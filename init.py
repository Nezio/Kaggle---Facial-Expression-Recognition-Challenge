import main

from win10toast import ToastNotifier

# configs
import configs.default as config_default
import configs.default2 as config_default2


def init():
    
    main.run(config_default)
    #main.run(config_default2)



    # notify Windows that all tasks are done
    toaster = ToastNotifier()
    toaster.show_toast("Facial Expression Recognition","All tasks completed!")





if __name__ == "__main__":
    init()