import time

def error(msg):
    print("[ERROR]:"+msg)
    exit(-1)


def log(msg):
    if isinstance(msg, str)==False:
        print(msg)
    else:
        print("[LOG] "+ time.strftime("%Y-%m-%d %H:%M:%S")+ ": "+msg)


def Process(msg,pos_end):
    if pos_end==True:
        print("\r[LOG]:"+msg)
    else:
        print("\r[LOG] :" +msg,end='')


def Warning(msg):
    print("[WARNING]:"+msg)


def log_loss(model, epoch ,train_loss , vali_loss):  


    print("[LOG] "+ time.strftime("%Y-%m-%d %H:%M:%S")+ 
          " [{0}] Epoch: {1} | Train Loss: {2:.6f} Vali Loss: {3:.6f}".
            format(model, epoch, train_loss, vali_loss))
    
def log_mse(model, epochs, mse):
    print("[LOG] "+ time.strftime("%Y-%m-%d %H:%M:%S")+ 
          " [{0} - {1}] Re-Normed Test Loss : {2:.6f}".
            format(model, epochs, mse))