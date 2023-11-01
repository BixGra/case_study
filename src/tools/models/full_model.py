from src.tools.models.action_model import *
from src.tools.models.duration_model import *
from src.tools.models.norm_model import *

action_model = ActionModel()
action_model.load("./models/ActionModel_1698809867")

duration_model = DurationModel()

norm_model = NormModel()


class FullModel:
    def __init__(self):
        pass

    @staticmethod
    def predict(predict_duration):
        tensors = [torch.randn([9]) for _ in range(action_model.sequence_length)]
        total_duration = 0
        output = []
        while total_duration < predict_duration:
            x = torch.unsqueeze(
                torch.cat(list(map(lambda t: torch.unsqueeze(t, dim=0), tensors[-action_model.sequence_length:])),
                          dim=0).type(torch.float), dim=0)
            action, action_tensor = action_model.predict(x)
            action_duration = duration_model.predict(action)
            norms = norm_model.predict(action_duration, action)
            output.append({"label": action, "norm": norms})
            total_duration += action_duration / (
                    50 * 60
            )  # action duration is 50Hz / predict duration is in minutes
        return output
