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



def init():

    main.run(C3_filter16)
    main.run(C3_filter16)
    
    main.run(C3_filter32)
    main.run(C3_filter32)
    
    main.run(C3_filter128)
    main.run(C3_filter128)
    

    


    # notify Windows that all tasks are done
    toaster = ToastNotifier()
    toaster.show_toast("Facial Expression Recognition","All tasks completed!")





if __name__ == "__main__":
    init()