import main

from win10toast import ToastNotifier

# configs
import configs.default as config_default

def init():
    
    main.run(config_default)




    # notify Windows that all tasks are done
    toaster = ToastNotifier()
    toaster.show_toast("Facial Expression Recognition","All tasks completed!")






if __name__ == "__main__":
    init()