"""Minimal, checkpoint-compatible MimicIK network and two-call decoder."""

from copy import deepcopy

import torch
from torch import nn

from ._nn.encoder import MLPEncoder
from ._nn.sudeepdit import SudeepDiT


class FlowMap(nn.Module):
    """Shared network interface used by MIP supervision and the FK auxiliary loss."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def get_velocity(self, time, action, condition):
        return self.net(action, time, time, condition)[0]

    def forward(self, start, end, action, condition=None):
        prediction = self.net(action, start, end, condition)[0]
        shape = (len(start),) + (1,) * (action.ndim - 1)
        return action + (end - start).reshape(shape) * prediction


class MimicIKModel(nn.Module):
    """SudeepDiT plus the original history-flattening state encoder.

    ``config`` uses the research code's ``task``, ``network`` and
    ``optimization`` sections. Only the released single-robot state model is
    supported; alternate encoders or morphology/identity features are rejected.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = deepcopy(config)
        task = config["task"]
        network = config["network"]
        optimization = config["optimization"]
        self.dof = int(task["act_dim"])
        self.obs_dim = int(task["obs_dim"])
        self.obs_steps = int(task["obs_steps"])
        self.horizon = int(task["horizon"])
        self.t_two_step = float(optimization["t_two_step"])
        if self.dof not in (6, 7) or self.obs_dim != self.dof + 14:
            raise ValueError(
                "Expected a single 6/7-DoF robot with q/current EEF/target EEF"
            )
        if self.obs_steps < 1 or self.horizon != 1:
            raise ValueError(
                "MimicIK requires positive observation history and horizon=1"
            )
        if task.get("obs_type", "state") != "state":
            raise ValueError("Only state observations are supported")
        if task.get("env_name", "ik_delta_joint") != "ik_delta_joint":
            raise ValueError("Only delta-joint IK checkpoints are supported")
        if network.get("network_type", "sudeepdit") != "sudeepdit":
            raise ValueError("Only the released SudeepDiT model is supported")
        if network.get("encoder_type", "mlp") != "mlp":
            raise ValueError(
                "Only the original history-flattening MLP encoder is supported"
            )
        if network.get("timestep_emb_type", "positional") != "positional":
            raise ValueError("Only positional time embeddings are supported")
        if network.get("disable_time_embedding", False):
            raise ValueError(
                "Time-disabled single-pass models are not MimicIK checkpoints"
            )
        if optimization.get("loss_type", "mip") != "mip":
            raise ValueError("Only MIP checkpoints are supported")
        if not 0 < self.t_two_step < 1:
            raise ValueError("t_two_step must lie strictly between zero and one")
        width = int(network["emb_dim"])
        depth = int(network["num_layers"])
        heads = int(network["n_heads"])
        encoder_depth = int(network.get("num_encoder_layers", 2))
        if (
            width < 8
            or width % 4
            or heads < 1
            or width % heads
            or min(depth, encoder_depth) < 1
        ):
            raise ValueError("Invalid model width, heads, or layer count")
        # Preserve construction order and parameter initialization of TrainingAgent.
        self.flow_map = FlowMap(
            SudeepDiT(
                act_dim=self.dof,
                Ta=self.horizon,
                obs_dim=width,
                To=self.obs_steps,
                d_model=width,
                n_heads=heads,
                depth=depth,
                dropout=float(network.get("dropout", 0.0)),
                timestep_emb_type="positional",
            )
        )
        self.encoder = MLPEncoder(
            obs_dim=self.obs_dim,
            emb_dim=width,
            To=self.obs_steps,
            hidden_dims=[width] * encoder_depth,
            dropout=float(network.get("encoder_dropout", 0.0)),
        )

    def predict_normalized(
        self, obs: torch.Tensor, endpoint: str = "final"
    ) -> torch.Tensor:
        """Return a complete normalized delta-q estimate, never a residual sum."""
        if endpoint not in ("first", "final"):
            raise ValueError("endpoint must be 'first' or 'final'")
        if (
            obs.ndim != 3
            or tuple(obs.shape[1:]) != (self.obs_steps, self.obs_dim)
            or len(obs) == 0
        ):
            raise ValueError(
                f"Expected nonempty [batch,{self.obs_steps},{self.obs_dim}] observations"
            )
        if not torch.isfinite(obs).all():
            raise ValueError("Observations must be finite")
        condition = self.encoder(obs, None)
        zeros = obs.new_zeros((len(obs), self.horizon, self.dof))
        first = self.flow_map.get_velocity(obs.new_zeros(len(obs)), zeros, condition)
        if endpoint == "first":
            return first
        time = obs.new_full((len(obs),), self.t_two_step)
        return self.flow_map.get_velocity(time, first, condition)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.predict_normalized(obs)
