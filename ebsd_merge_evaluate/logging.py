from datetime import datetime
import os

def initTimestamp(self):
    self.now = datetime.now()
    self.nowstr = (
            str(self.now.date())
            + "_"
            + "{:02}".format(self.now.hour)
            + "h_"
            + "{:02}".format(self.now.minute)
            + "m_"
            + "{:02}".format(self.now.second)
            + "s"
    )


def createLogFileMerge(self):
    self.logfile_merge_path = os.path.join(
        "tmp", self.nowstr + "_logfile_merging.log"
    )
    logfile = open(self.logfile_merge_path, "w")
    # logfile.writelines('_Header_\n')
    logfile.writelines(
        "Logfile created: "
        + str(self.now.date())
        + " at "
        + str(self.now.hour)
        + ":"
        + str(self.now.minute)
        + ":"
        + str(self.now.second)
        + "\n\n"
    )
    logfile.close()


def createLogFileEval(self):
    self.logfile_eval_path = os.path.join(
        "tmp", self.nowstr + "_logfile_evaluation.log"
    )
    logfile = open(self.logfile_eval_path, "w")
    # logfile.writelines('_Header_\n')
    logfile.writelines(
        "Logfile created: "
        + str(self.now.date())
        + " at "
        + str(self.now.hour)
        + ":"
        + str(self.now.minute)
        + ":"
        + str(self.now.second)
        + "\n\n"
    )
    logfile.close()


def logNewHead(
        self,
        filepath: str,
        title: str,
):
    if not os.path.isfile(filepath):
        print("Error: Logfile not initialized!")
    else:
        linewidth = 40
        logfile = open(filepath, "a")
        logfile.writelines(
            "-" * (linewidth - int(len(title) / 2))
            + " "
            + title
            + " "
            + "-" * (linewidth - int(len(title) / 2))
            + "\n\n"
        )
        logfile.close()


def logNewLine(
        self,
        filepath: str,
        text: str,
):
    if not os.path.isfile(filepath):
        print("Error: Logfile not initialized!")
    else:
        logfile = open(filepath, "a")
        logfile.writelines(text + "\n")
        logfile.close()


def logNewSubline(
        self,
        filepath: str,
        text: str,
):
    self.logNewLine(filepath, " - " + text)