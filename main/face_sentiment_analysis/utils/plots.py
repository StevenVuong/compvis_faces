import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_csv_plot(csv_path: str, save_plot: bool = False):
    """
    Loads csv and plot
    """
    history_df = pd.read_csv(csv_path)

    acc = history_df["acc"].values
    loss = history_df["loss"].values
    val_acc = history_df["val_acc"].values
    val_loss = history_df["val_loss"].values

    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    
    # Plots for acc
    axs[0].plot(acc)
    axs[0].plot(val_acc)

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, (len(acc) + 1),(len(acc) / 10)))
    axs[0].legend(['train', 'val'], loc='best')

    # Plots for loss
    axs[1].plot(loss)
    axs[1].plot(val_loss)

    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, (len(loss) + 1), (len(loss) / 10)))
    axs[1].legend(['train','val'], loc='best')
    
    if save_plot:
        fig.savefig('./plot.png')

    plt.show()


def main():
    read_csv_plot("./history.csv", save_plot=True)

if __name__=="__main__":
    main()