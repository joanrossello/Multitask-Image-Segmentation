import torch
import torch.nn as nn


class MTL_uncertain(nn.Module):
    """
    Implements learning the losses using homoscedastic uncertanties.
    task_flags : indicating which tasks are being done in order seg, class, recon
    task_order : order in which the given model parameter gives outputs
    i.e. tuple (pos_of_seg, pos_of_class, [optional] pos_of_recon)

    """

    def __init__(self, model, task_flags, task_order):
        super(MTL_uncertain, self).__init__()
        self.model = model

        self.task_order = task_order
        self.seg_flag, self.class_flag, self.recon_flag = task_flags

        self.seg_criterion = nn.BCEWithLogitsLoss()
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss()

        if not self.recon_flag:
            self.logsigmas = nn.Parameter(torch.zeros((sum(task_flags))))
        else:
            self.logsigmas = nn.Parameter(torch.cat((torch.zeros(sum(task_flags) - 1),torch.tensor(4).unsqueeze(0))))

    def forward(self, input, target_seg, target_class):
        outputs = self.model(input)

        # reconstruction target is image itself
        target_recon = input

        assert (self.seg_flag == True)
        assert (self.class_flag == True)

        pred_seg = outputs[self.task_order[0]]
        pred_class = outputs[self.task_order[1]]

        pred_recon = None
        if (self.recon_flag):
            pred_recon = outputs[self.task_order[2]]

        ### not sure abt 1*x, for cross-entropy transformation

        cse_trans = lambda x: 1 * x

        ## regression transformation
        reg_trans = lambda x: 0.5 * x

        seg_loss = self.seg_criterion(pred_seg, target_seg)
        total_loss = torch.sum(cse_trans(seg_loss) * torch.exp(-2 * self.logsigmas[0]) + self.logsigmas[0])

        class_loss = self.class_criterion(pred_class, target_class)
        total_loss += torch.sum(cse_trans(class_loss) * torch.exp(-2 * self.logsigmas[1]) + self.logsigmas[1])

        recon_loss = None
        if (self.recon_flag):
            recon_loss = self.recon_criterion(pred_recon, target_recon)
            total_loss += torch.sum(reg_trans(recon_loss) * torch.exp(-2 * self.logsigmas[2]) + self.logsigmas[2])

        return (pred_seg, pred_class, pred_recon,
                seg_loss, class_loss, recon_loss,
                self.logsigmas.data.tolist(), total_loss)



