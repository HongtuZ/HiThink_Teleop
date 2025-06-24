from .agilex_piper import AgilexPiper

def NoneRobot(*args, **kwargs):
    return None

RobotComponents = {
    'agilex_piper': AgilexPiper,
    None: NoneRobot
}