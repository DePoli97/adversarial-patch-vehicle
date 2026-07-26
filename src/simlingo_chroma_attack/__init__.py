"""White-box adversarial patch against SimLingo (`simlingo_simlingo`).

SimLingo is a vision-language-action agent (Mini-InternVL2-1B: an InternViT-300M
vision encoder feeding a LoRA-tuned Qwen2-0.5B LLM). It emits no objectness and
no distance, so the detector-style losses used in `yolo_chroma_attack` and
`clip_chroma_attack` have nothing to attach to — which is why attacking it
through an external detector never transferred.

The attack surface is the waypoints. `DrivingAdaptor.get_predictions`
(adaptors.py:163-181) runs `self.heads[t](feature).cumsum(1)`: the heads emit
per-step deltas, the cumsum makes them absolute ego-relative coordinates. The
agent then turns two of those waypoints into throttle
(`agent_simlingo.control_pid`, agent_simlingo.py:858-863):

    one_second   = carla_fps // (wp_dilation * data_save_freq) = 20 // 5 = 4
    half_second  = 2
    desired_speed = ||wp[0] - wp[2]|| * 2.0          # m/s
    brake         = desired_speed < 0.4  or  ego_speed / desired_speed > 1.1

So there is exactly ONE lever: push the waypoints farther forward and the car
believes it may accelerate; it is the same knob that decides whether it brakes
at all. `SimlingoSpeedUpLoss` maximises that quantity.

Modules
-------
`simlingo_model.py` — everything that couples to the PCLA checkpoint: standalone
    load of `DrivingModel`, a differentiable re-implementation of the agent's
    image preprocessing, prompt construction, and the single-forward
    (`predict_language=False`) path that keeps gradients alive.
`simlingo_loss.py` — `SimlingoSpeedUpLoss`, same `(image, target_bbox) ->
    (loss, info)` contract as `YoloHideLoss`, so it drops into the shared
    training skeleton.
`train.py` — the Adam-over-pixels loop, reusing `yolo_chroma_attack`'s
    dataset / differentiable renderer / EoT / TV.
"""
