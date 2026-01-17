from roboflow import Roboflow
rf = Roboflow(api_key="hWWijfYS3fWAT41yIrdw")
project = rf.workspace("test-uclkg").project("nutri-scanner-pkm")
version = project.version(1)
dataset = version.download("yolov8")
                