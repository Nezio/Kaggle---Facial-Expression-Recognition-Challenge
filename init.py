import main

from win10toast import ToastNotifier

# configs
import configs.default as config_default
import configs.conv_x2 as config_conv_x2
import configs.conv_x3 as config_conv_x3
import configs.conv_x4 as config_conv_x4
import configs.C3_filter16 as C3_filter16
import configs.C3_filter32 as C3_filter32
import configs.C3_filter128 as C3_filter128

import configs.C3_dense64 as C3_dense64
import configs.C3_dense128 as C3_dense128
import configs.C3_dense512 as C3_dense512
import configs.C3_dense1024 as C3_dense1024

import configs.C3_dropout15 as C3_dropout15
import configs.C3_dropout35 as C3_dropout35
import configs.C3_dropout45 as C3_dropout45
import configs.C3_dropout55 as C3_dropout55
import configs.C3_dropout65 as C3_dropout65

import configs.C3F32_batch16 as C3F32_batch16
import configs.C3F32_batch32 as C3F32_batch32
import configs.C3F32_batch128 as C3F32_batch128
import configs.C3F32_batch256 as C3F32_batch256

import configs.C3F32_lr0_00001 as C3F32_lr0_00001
import configs.C3F32_lr0_0001 as C3F32_lr0_0001
import configs.C3F32_lr0_01 as C3F32_lr0_01
import configs.C3F32_lr0_1 as C3F32_lr0_1

import configs.C3F32_fullDataset as C3F32_fullDataset


def init():

    main.run(C3F32_fullDataset)
    


    # notify Windows that all tasks are done
    toaster = ToastNotifier()
    toaster.show_toast("Facial Expression Recognition","All tasks completed!")





if __name__ == "__main__":
    init()