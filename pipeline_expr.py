from dataclasses import dataclass

import torch
import tyro
from torch import optim

from losses import tv_loss
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils import LOGGER, AorB
from utils_st import (calc_mean_std, cWCT,
                            exact_feature_distribution_matching, wct)


@dataclass
class ExprConfig(OptimzeBaseConfig):
    """Directly use target features to reconstruct."""

    name: str = 'explore'
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception"""
    # layers: str = "1"
    layers: str = "1, 6, 11, 18, 25"
    # layers: str = "25, 27, 29, 22, 20, 18, 15, 13, 11, 8, 6, 3, 1"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    max_iter: int = 300
    save_process: bool = False
    use_tv_reg: bool = False
    save_init: bool = False
    exact_resize: bool = True
    use_tensorboard: bool = False


class ExprPipeline(OptimzeBasePipeline):
    def __init__(self, config: ExprConfig) -> None:
        super().__init__(config)
        self.config = config
        self.kl_loss = torch.nn.KLDivLoss()
    
    def add_extra_infos(self):
        return [f'rec_efdm'] + AorB(self.config.use_tv_reg, '10tvloss')
    
    def add_extra_file_infos(self):
        return ['']
    
    def common_optimize_process(self, Ic, Is):
        opt_img = Ic.data.clone()
        opt_img = opt_img.to(self.device)

        # save init image
        self.save_image(opt_img, 'init', verbose=True)

        opt_img.requires_grad = True

        # get target features
        _, Is_features = self.model(Is)
        Ic_41, Ic_features = self.model(Ic)

        # !construct target features
        target_features = []

        for cur, style in zip(Ic_features, Is_features):
            c_mean, c_std = calc_mean_std(cur)
            s_mean, s_std = calc_mean_std(style)

            #! AdaIN
            # target_features.append(((cur - c_mean)/c_std) * s_std + s_mean)

            #! WCT
            # target_features.append(cWCT(cur, style))

            #! efdm
            target_features.append(exact_feature_distribution_matching(cur, style))

        optimizer = optim.LBFGS([opt_img], lr=1)

        n_iter = [0]
        while n_iter[0] <= self.config.max_iter:
            def closure():
                loss = 0
                optimizer.zero_grad()

                Io_41, Io_feats = self.model(opt_img)

                # content_loss = torch.mean((Io_41 - content_features) ** 2)
                
                for cur, style in zip(Io_feats, target_features):
                    loss += torch.mean((cur - style) ** 2)
                
                if self.config.use_tv_reg:
                    total_variance_loss = tv_loss(opt_img)
                    loss += 10 * total_variance_loss

                loss.backward()

                n_iter[0] += 1

                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    self.save_image(opt_img, str(n_iter[0]//self.config.show_iter), verbose=True)
                return loss
            
            optimizer.step(closure)
        
        # save final image
        self.save_image(opt_img, 'result')
        
    
if __name__ == '__main__':
    args = tyro.cli(ExprConfig)
    pipeline = ExprPipeline(args)
    content_dir='data/content/'
    style_dir='data/nnst_style/'
    # for c in os.listdir(content_dir):
    #     for s in os.listdir(style_dir):
    #         pipeline(
    #             content_dir + c,
    #             style_dir + s
    #         )
    # exit()

    # for s in os.listdir(style_dir):
    #     pipeline(
    #         'data/content/sailboat.jpg',
    #         style_dir + s
    #     )
    
    # for c in os.listdir(content_dir)[:5]:
    #     pipeline(
    #         content_dir + c,
    #         'data/nnst_style/S4.jpg'
    #     )

    pipeline(
        # 'data/content/C1.png', 
        'data/content/sailboat.jpg', 
        # 'data/style/17.jpg'
        'data/styles/0.jpg'
    )
    