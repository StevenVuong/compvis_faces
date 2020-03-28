import matplotlib.pyplot as plt
import pandas as pd

def read_csv_plot(csv_path: str):
    """
    Loads csv and plot
    """
    history_df = pd.read_csv(csv_path)

    acc = history_df["Acc"]
    loss = history_df["Loss"]
    val_acc = history_df["Val_acc"]
    val_loss = history_df["Val_loss"]

    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # Plots for acc
    axs[0].plot(range(1, len(acc) + 1), acc])
    axs[0].plot(range(1, len(val_acc) + 1), val_acc)

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(acc) + 1),len(acc]) / 10)
    axs[0].legend(['train'], loc='best')

    # Plots for loss
    axs[1].plot(range(1, len(loss) + 1), loss)
    axs[1].plot(range(1, len(val_loss) + 1), val_loss)

    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(loss) + 1), len(loss) / 10)
    axs[1].legend(['train'], loc='best')
    
    # fig.savefig('plot.png')
    plt.show()


def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # Summarize history for accuracy
    axs[0].plot(
        range(1, len(model_history.history['accuracy'])+1),
        model_history.history['accuracy'])
    axs[0].plot(
        range(1,len(model_history.history['val_acc'])+1),
        model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(
        1,
        len(model_history.history['accuracy'])+1),
        len(model_history.history['accuracy'])/10)
    axs[0].legend(['train'], loc='best')
    # Summarize history for loss
    axs[1].plot(range(
        1,
        len(model_history.history['loss'])+1),
        model_history.history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1,
        len(model_history.history['loss'])+1),
        len(model_history.history['loss'])/10)
    axs[1].legend(['train'], loc='best')
    # fig.savefig('plot.png')
    plt.show()

    ## Add Saveplot


def main():
    read_csv_plot("./history.csv")

if __name__=="__main__":
    main()