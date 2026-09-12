# trials & stim labels

TRIALS = ["NHM", "SHM", "LHM"]
BASELINE_PERIODS = [(f"Baseline_Start_{T}", f"Task_Start_0_{T}") for T in TRIALS] + [("Baseline_End", "Experiment_End")]

TASK_START_IT = "Task_Start_{i}_{T}"
REST_IT = "Rest_{i}_{T}"

MOTION_START_IT = "Task_Motion_Start_{i}_{T}"
MOTION_END_IT = "Task_Motion_End_{i}_{T}"

NUM_BLOCKS = 8