from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
from psychopy.visual.grating import GratingStim
import numpy as np
import random
import os


def load_exp_info(filename: str):
    try:
        exp_info = fromFile(filename)
    except:
        exp_info = {"observer": "mkh", "ref_orientation": 0}
    exp_info["date"] = data.getDateStr()
    return exp_info


class Logger(object):
    def __init__(self, filename: str, verbose: bool = False):
        self.log_file = open(filename, 'w', encoding='utf-8')
        self.verbose = verbose
    
    def log(self, content: str):
        if self.verbose:
            print(content)
        self.log_file.write(content + '\n')
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def main():
    # get info
    record_root = "./record"
    params_filename = os.path.join(record_root, "last_params.pickle")
    exp_info = load_exp_info(params_filename)

    # present a dialogue
    dlg = gui.DlgFromDict(exp_info, title="Simple JND Exp", fixed=["date"])
    if dlg.OK:
        toFile(params_filename, exp_info)
    else:
        core.quit()
    
    # log
    save_dir = os.path.join(record_root, exp_info["observer"])
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, exp_info["date"] + ".csv"), verbose=False)
    logger.log("target_side,ori_increment,correct")

    # staircase handler
    staircase = data.StairHandler(startVal=20.0, stepType='db', stepSizes=[8, 4, 4, 2],
                                  nUp=1, nDown=3, nTrials=1)
    
    # create window and stimulation
    window = visual.Window(size=[800, 600], allowGUI=True, monitor="testMonitor", units="deg")
    foil = GratingStim(window, sf=1, size=4, mask="gauss", ori=exp_info["ref_orientation"]) # sf: Spatial Frequency
    target = GratingStim(window, sf=1, size=4, mask="gauss", ori=exp_info["ref_orientation"])
    fixation = GratingStim(window, color=-1, colorSpace="rgb", tex=None, mask="circle", size=0.2)

    # clocks
    global_clock = core.Clock()
    trial_clock = core.Clock()

    # display instructions and wait
    msg1 = visual.TextStim(window, pos=[0, +3], text="Hit a key when ready.")
    msg2 = visual.TextStim(window, pos=[0, -3], text=f"Then press left or right to identify the {exp_info['ref_orientation']:.1f} deg probe.")
    msg1.draw()
    msg2.draw()
    fixation.draw()
    window.flip()

    event.waitKeys()

    for increment in staircase:
        target_side = random.choice([-1, 1])
        foil.setPos([-5 * target_side, 0])
        target.setPos([5 * target_side, 0])
        foil.setOri(exp_info['ref_orientation'] + increment)

        foil.draw()
        target.draw()
        fixation.draw()
        window.flip()

        core.wait(0.5)

        # blank screen
        fixation.draw()
        window.flip()

        # get response
        response = None
        while response is None:
            keys = event.waitKeys()
            for key in keys:
                if key == "left":
                    response = -target_side
                elif key == "right":
                    response = target_side
                elif key in ['q', "escape"]:
                    core.quit()
            event.clearEvents()
        
        staircase.addData(response)
        logger.log(f"{target_side},{increment:.3f},{response}")
        core.wait(1)
    
    logger.close()
    staircase.saveAsPickle(os.path.join(save_dir, exp_info["date"]))
    
    print(f"reversals: {staircase.reversalIntensities}")
    approx_threshold = np.average(staircase.reversalIntensities[-6:])
    feedback = visual.TextStim(window, pos=[0, +3], text=f"mean of final 6 reversals: {approx_threshold:.3f}")
    feedback.draw()
    fixation.draw()
    window.flip()
    event.waitKeys()

    window.close()
    core.quit()


if __name__ == "__main__":
    main()