# A100 pilot runbook

Pilot archives write only to
`classA_pilot_outputs/production_10sample_v4_occupied_frame_cycle_resolved` and never satisfy
production automatically.  Pilot geometry also fixes `Nx=20`.  Run validation and one
representative shard from each kernel class, then replace planning estimates with the
measured synchronized-GPU time recorded in its manifest.  Pilot data are diagnostic and
must not be pooled silently with the 10-trajectory production ensemble.
