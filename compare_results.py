import re
import cv2
import matplotlib.pyplot as plt
from utils import *
from utils_st import *
import os
import matplotlib.pyplot as plt

# 设置西文字体为新罗马字体
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
#     "mathtext.fontset":'stix',
}
rcParams.update(config)


def prepare_image(path):
    return cv2.resize(cv2.imread(path), (512, 512))


def crop_features_image():
    targets = [
        "results/FeatureImages/Content.png",
        "results/FeatureImages/Style.png",
        "results/FeatureImages/AdaIN_target.png",
        "results/FeatureImages/EFDM_target.png",
        "results/FeatureImages/WCT_Target.png",
        "results/FeatureImages/OptResult.png",
    ]
    indices = [
        [1, 1],
        [4, 5]
    ]
    for target in targets:
        img = Image.open(target)
        tag = target.split('/')[-1][:-4]
        for inds in indices:
            cropped = img.crop((inds[0] * 512, inds[1] * 512, (inds[0]+1) * 512, (inds[1]+1) * 512  - 1))  # (left, upper, right, lower)
            cropped.save(f"./results/FeatureImages/{tag}_{inds[0]}_{inds[1]}.png")
    print('done!!!')


if __name__ == '__main__':
    CONTENT_DIR = 'data/contents/'
    STYLE_DIR = 'data/styles/'
    OUTPUT_DIRS = []
    task = 5
    all_images = []
    titles = []
    titles_bottom = []

    # Figure 1
    if task == 0:
        OUTPUT_DIRS.append('results/TestMetrics/Test_WCT/')
        OUTPUT_DIRS.append('results/TestMetrics/Test_ArtFlow-WCT/')
        # OUTPUT_DIRS.append('results/TestMetrics/Test_ArtFlow-AdaIN/')
        OUTPUT_DIRS.append('results/TestMetrics/Test_AdaAttn/')
        OUTPUT_DIRS.append('results/TestMetrics/pama/')
        OUTPUT_DIRS.append('results/TestMetrics/Test_StyleTr2/')
        OUTPUT_DIRS.append('results/TestMetrics/Test_NNST/')
        # OUTPUT_DIRS.append(r'results/TestMetrics/nnst/common_25-18-11-6-1_remd/')
        OUTPUT_DIRS.append('results/TestMetrics/Test_STROTSS/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs64/')
        titles = ['Content', 'Style', 'WCT', 'ArtFlow_WCT', 'AdaAttn', 'PAMA', 'StyleTr2', 'NNST', 'STROTSS', 'Ours']
        # content style pairs
        Ic_lst = ['4', '5', '6', '13', '0', '1', '15', '12']
        Is_lst = ['14', '12', '3', '15', '7', '9', '0', '10']

    # Figure 2
    if task == 1:
        # OUTPUT_DIRS.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_mean/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_l2/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_adain/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_efdm/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_gatysdivc/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_cov/')
        # OUTPUT_DIRS.append(r'results/TestMetrics/nnfm/common_25-18-11-6-1_FS_nn/')
        OUTPUT_DIRS.append(r'results/TestMetrics/nnfm/common_25-18-11-6-1_FS_remd/')
        # OUTPUT_DIRS.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_cov-1pyr-300/')
        # OUTPUT_DIRS.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs64-1pyr-300/')
        OUTPUT_DIRS.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs64/')
        titles = ['Content', 'Style', 'L2', 'AdaIN', 'Efdm', 'Gatys', 'Cov', 'Remd', 'Ours']
        Ic_lst = ['8', '7', '14']
        Is_lst = ['11', '1', '13']

    # search target results of methods
    if task == 0 or task == 1:
        for Ic, Is in zip(Ic_lst, Is_lst):
            content_path = CONTENT_DIR + (f'{Ic}.png' if str(Ic) in '1, 2, 6, 7, 8, 9' else f'{Ic}.jpg')
            style_path = STYLE_DIR + f'{Is}.jpg'

            images = []
            images += [prepare_image(content_path), prepare_image(style_path)]

            for i, output_dir in enumerate(OUTPUT_DIRS):
                output = ''
                for file in os.listdir(output_dir):
                    matches = re.findall(r'\d+', file)
                    if output_dir == 'results/TestMetrics/Test_AdaAttn/':
                        content = matches[1]
                        style = matches[0]
                    else:
                        content = matches[0]
                        style = matches[1]
                    if Ic == content and Is == style:
                        output = file
                        break

                # print(output)
                img = cv2.imread(output_dir + output)
                images += [img]

            all_images.append(images)
            rows = len(Ic_lst)
            cols = len(OUTPUT_DIRS) + 2

    if task == 2:
        titles = ['Content&Style', '1 scale', '2 scales', '3 scales', '4 scales', '5 scales']
        row1, row2 = [], []
        row1.append(prepare_image(CONTENT_DIR + '11.jpg'))
        row1.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_pyr1/11_0_result.png'))
        row1.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_pyr2/11_0_result.png'))
        row1.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_pyr3/11_0_result.png'))
        row1.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_pyr4/11_0_result.png'))
        row1.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_pyr5/11_0_result.png'))
        
        row2.append(prepare_image(STYLE_DIR + '0.jpg'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_remd_pyr1/11_0_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_remd_pyr2/11_0_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_remd_pyr3/11_0_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_remd_pyr4/11_0_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_remd_pyr5/11_0_result.png'))

        all_images.append(row1)
        all_images.append(row2)
        rows, cols = len(all_images), len(row1)
    
    if task == 3:
        titles = ['Content', 'Style', 'Cov+Remd']
        titles_bottom = ['Cov', 'Remd', 'Ours']
        row1 = []
        row2 = []
        row1.append(prepare_image(CONTENT_DIR + '8.png'))
        row1.append(prepare_image(STYLE_DIR + '2.jpg'))
        row1.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_cov+remd2pyr/8_2_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_pyr2/8_2_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_remd_pyr2/8_2_result.png'))
        row2.append(prepare_image('results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs64/8_2_result.png'))

        all_images.append(row1)
        all_images.append(row2)
        rows = len(all_images)
        cols = len(row1)
    
    if task == 4:
        titles = ['Content', 'Style', 'AdaIN', 'EFDM', 'WCT', 'Opt Result']
        # titles_bottom = ['Cov', 'Remd', 'Ours']
        row1 = []
        row2 = []
        row1.append(prepare_image("results/FeatureImages/Content_1_1.png"))
        row1.append(prepare_image("results/FeatureImages/Style_1_1.png"))
        row1.append(prepare_image("results/FeatureImages/AdaIN_target_1_1.png"))
        row1.append(prepare_image("results/FeatureImages/EFDM_target_1_1.png"))
        row1.append(prepare_image("results/FeatureImages/WCT_Target_1_1.png"))
        row1.append(prepare_image("results/FeatureImages/OptResult_1_1.png"))

        row2.append(prepare_image("results/FeatureImages/Content_4_5.png"))
        row2.append(prepare_image("results/FeatureImages/Style_4_5.png"))
        row2.append(prepare_image("results/FeatureImages/AdaIN_target_4_5.png"))
        row2.append(prepare_image("results/FeatureImages/EFDM_target_4_5.png"))
        row2.append(prepare_image("results/FeatureImages/WCT_Target_4_5.png"))
        row2.append(prepare_image("results/FeatureImages/OptResult_4_5.png"))

        all_images.append(row1)
        all_images.append(row2)
        rows = len(all_images)
        cols = len(row1)
    
    if task == 5:
        titles = ['Content', 'Style', 'AdaIN', 'EFDM', 'WCT']
        # titles_bottom = ['Cov', 'Remd', 'Ours']
        row1 = []
        row2 = []
        row1.append(prepare_image(CONTENT_DIR + '0.jpg'))
        row1.append(prepare_image(STYLE_DIR + '0.jpg'))
        row1.append(prepare_image("results/explore/common_1-6-11-18-25_rec_adain/sailboat_S4__result.png"))
        row1.append(prepare_image("results/explore/common_1-6-11-18-25_rec_efdm/sailboat_S4__result.png"))
        row1.append(prepare_image("results/explore/common_1-6-11-18-25_rec_cWCT/sailboat_S4__result.png"))

        # row2.append(prepare_image("results/FeatureImages/Content_4_5.png"))
        # row2.append(prepare_image("results/FeatureImages/Style_4_5.png"))
        # row2.append(prepare_image("results/FeatureImages/AdaIN_target_4_5.png"))
        # row2.append(prepare_image("results/FeatureImages/EFDM_target_4_5.png"))
        # row2.append(prepare_image("results/FeatureImages/WCT_Target_4_5.png"))
        # row2.append(prepare_image("results/FeatureImages/OptResult_4_5.png"))

        all_images.append(row1)
        # all_images.append(row2)
        rows = len(all_images)
        cols = len(row1)

    
    # 布局
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.2))

    if len(axes.shape) == 1:
        axes = axes.reshape(1, -1)
    
    print(axes.shape)

    for i in range(rows):
        for j in range(cols):
            # opencv
            pic = all_images[i][j]
            pic = pic[..., ::-1]
            
            axes[i,j].imshow(pic)
            if i == 0:
                axes[i,j].set_title(titles[j])
            if i == rows - 1 and len(titles_bottom) > 0:
                axes[i,j].set_title(titles_bottom[j], y=-0.3)
            axes[i,j].axis('off')    

    plt.subplots_adjust(
        left=0.10,
        bottom=0.01,
        right=0.840,
        top=0.86,
        # top=0.85,
        wspace=0,
        hspace=0.01
    )

    # plt.text(x=-1.1,y=1.3, s='Content', color='black', rotation=90)
    plt.savefig(f'fig_0{task+1}.jpg', bbox_inches='tight', dpi=300)
    plt.show()