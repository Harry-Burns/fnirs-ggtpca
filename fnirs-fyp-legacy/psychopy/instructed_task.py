import argparse
import time

from psychopy import visual, core, event


parser = argparse.ArgumentParser(prog='fNIRS Task Recording Instructor')
parser.add_argument('--uselsl', action='store_true')

WINDOW_RES = (1280,720)

# --- Test Parameters
BASELINE_DUR = 20.0 # Initial & final baseline duration

TASK_DUR = 10.0 # Task duration
REST_DUR = 20.0 # Rest duration

N_BLOCKS = 8 # Number of blocks per trial

TRIALS = ["NHM", "SHM", "LHM"]



# Motion Params (seconds)
SHM_MOTION_DUR = 1.0 # SHM motion stim duration
LHM_MOTION_DUR = 1.0 # LHM motion stim duration
MOTION_PADDING = (4.0, 2.0) # Padding (before,after)

ELLIPSE_DUR = 2 # When to start countdown
# -----

if (TASK_DUR - max(ELLIPSE_DUR, MOTION_PADDING[0]) - MOTION_PADDING[1] - max(SHM_MOTION_DUR,LHM_MOTION_DUR)) <= 0: raise ValueError("Motion Schedule: Padding is too large for given task duration.")

import random
#random.seed(42) # REMOVE DURING TEST
init_time = time.time()


def motion_schedule(task_dur: float=10.0, motion_dur: float=1.0, padding: tuple[float,float]=MOTION_PADDING) -> tuple[int,int,int]:
    schedule_range = task_dur - sum(padding) - motion_dur
    motion_time = random.random() * schedule_range + padding[0] # time of start of motion
    return (motion_time, motion_dur, task_dur - motion_time - motion_dur)


def main(args):
    if args.uselsl: 
        print("LSL: Active")
        outlet = setup_lsl()
    else: 
        print("LSL: Inactive")

    win = visual.Window(WINDOW_RES, color="black", units="pix", fullscr=True) #monitor='testMonitor',screen=1)

    # -- defining window text
    instr_text = visual.TextStim(win, text="Finger Tapping Task\n\nEnsure you have read the instructions in the powerpoint.\n\nOnce you are ready, press ENTER to start.", color="white")
    test_text = visual.TextStim(win, text="* Test *", color="white")

    baseline_nhm_stim = [visual.TextStim(win, text="Baseline", color="white", height=70), visual.TextStim(win, text="Upcoming trial: NHM, prepare to tap...", color="white", height=20, pos=(0,-80))]
    baseline_shm_stim = [visual.TextStim(win, text="Baseline", color="white", height=70), visual.TextStim(win, text="Upcoming trial: SHM, prepare to tap...", color="white", height=20, pos=(0,-80))]
    baseline_lhm_stim = [visual.TextStim(win, text="Baseline", color="white", height=70), visual.TextStim(win, text="Upcoming trial: LHM, prepare to tap...", color="white", height=20, pos=(0,-80))]
    baseline_end_stim = [visual.TextStim(win, text="Baseline", color="white", height=70), visual.TextStim(win, text="Ending the recording", color="white", height=20, pos=(0,-50))]

    task_stim = visual.TextStim(win, text="TAP", color="cyan", height=80)
    task_shm_stim = visual.TextStim(win, text="TAP", color="yellow", height=80)
    task_lhm_stim = visual.TextStim(win, text="TAP", color="red", height=80)

    rest_stim = visual.TextStim(win, text="REST", color="white", height=70)
    end_text = visual.TextStim(win, text="Experiment Complete.\nThank you!", color="white")
    # --
    
    # -- Hselper functions
    def show_stim(stims,sample=None,wait_dur=None):
        if isinstance(stims, list):
            for stim in stims:  stim.draw()
        else:   stims.draw()  
        win.flip()
        if args.uselsl and sample is not None: outlet.push_sample([sample]); print(f"[{time.time()-init_time:.1f}] Pushing sample: {[sample]}")
        if wait_dur is not None: core.wait(wait_dur) 

    def ellipse_lead(stims, countdown_time=ELLIPSE_DUR, col="cyan", ellipse_num=2, spacing=80):
        if stims is None or not countdown_time: return
        
        if not isinstance(stims, list):
            stims = [stims]

        # Build the leading ellipses (hollow circles) along the bottom
        y=-100  # px between ellipses; px under centre 
        start_x = -spacing * (ellipse_num - 1) / 2

        ellipses = [visual.Circle(win, radius=10, pos=(start_x + i * spacing, y), lineColor=col, fillColor="None", lineWidth=3)   for i in range(ellipse_num)]
        step = countdown_time / max(1, ellipse_num)

        show_stim(stims + ellipses, sample=None, wait_dur=None)

        for i in range(ellipse_num):
            ellipses[i].fillColor = col
            show_stim(stims + ellipses, sample=None, wait_dur=step)
    # --

    # ---- Begin test execution

    # -- Instructions
    i = 0
    while True:
        show_stim(instr_text)
        key = event.waitKeys(keyList=['space', 'return'])[0]

        if key == 'return': break
        
        show_stim(test_text,sample=f'Test_Marker_{i}', wait_dur=1)
        i += 1
    # --

    # -- Trial Loop
    for i,t in enumerate(TRIALS):

        # Trial Start Baseline
        if t == "NHM":      show_stim(baseline_nhm_stim,sample=f'Baseline_Start_{t}',wait_dur=BASELINE_DUR)
        elif t == "SHM":    show_stim(baseline_shm_stim,sample=f'Baseline_Start_{t}',wait_dur=BASELINE_DUR)
        elif t == "LHM":    show_stim(baseline_lhm_stim,sample=f'Baseline_Start_{t}',wait_dur=BASELINE_DUR)

        for b in range(N_BLOCKS):
            if t == "NHM":
                show_stim(task_stim,sample=f'Task_Start_{b}_{t}',wait_dur=TASK_DUR)
                show_stim(rest_stim,sample=f'Rest_{b}_{t}',wait_dur=REST_DUR)

            elif t == "SHM":
                start_t,_,end_t = motion_schedule(task_dur=TASK_DUR, motion_dur=SHM_MOTION_DUR)

                show_stim(task_stim,sample=f'Task_Start_{b}_{t}',wait_dur=start_t-ELLIPSE_DUR)
                ellipse_lead(stims=task_stim, countdown_time=ELLIPSE_DUR, col="yellow")

                show_stim(task_shm_stim,sample=f'Task_Motion_Start_{b}_{t}',wait_dur=0)
                ellipse_lead(stims=task_shm_stim, countdown_time=SHM_MOTION_DUR, col="yellow", ellipse_num=10, spacing=30)

                show_stim(task_stim,sample=f'Task_Motion_End_{b}_{t}',wait_dur=end_t)

                show_stim(rest_stim,sample=f'Rest_{b}_{t}',wait_dur=REST_DUR)

            elif t == "LHM":
                start_t,_,end_t = motion_schedule(task_dur=TASK_DUR, motion_dur=LHM_MOTION_DUR)
                
                show_stim(task_stim,sample=f'Task_Start_{b}_{t}',wait_dur=start_t-ELLIPSE_DUR)
                ellipse_lead(stims=task_stim, countdown_time=ELLIPSE_DUR, col="red")

                show_stim(task_lhm_stim,sample=f'Task_Motion_Start_{b}_{t}',wait_dur=0)
                ellipse_lead(stims=task_lhm_stim, countdown_time=LHM_MOTION_DUR, col="red", ellipse_num=10, spacing=30)

                show_stim(task_stim,sample=f'Task_Motion_End_{b}_{t}',wait_dur=end_t)

                show_stim(rest_stim,sample=f'Rest_{b}_{t}',wait_dur=REST_DUR)
            else:
                raise RuntimeError(f"Unknown trial block reached: {t}")
    # --            

    # Final Baselines
    show_stim(baseline_end_stim,sample='Baseline_End',wait_dur=BASELINE_DUR)

    # End Experiment
    show_stim(end_text,sample='Experiment_End',wait_dur=4.0)
    # ---

    win.close()
    core.quit()




def setup_lsl():
    from pylsl import StreamInfo, StreamOutlet

    # name is arbitrary; type is a standard label; single-channel recording; no fixed recording ferquency; string data; arbitrary source 
    info = StreamInfo(name='InstructedTaskMarkers', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='12345')
    outlet = StreamOutlet(info)
    return outlet



if __name__ == "__main__":
    args = parser.parse_args()
    out = main(args)