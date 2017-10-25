from utility import dataset_creater
import matplotlib.pyplot as plt


def main():
    from impl.impl_networks import gen_plot_res
    X_train, y_train = dataset_creater.loadIT("train")
    X_valid, y_valid = dataset_creater.loadIT("valid")
    from utility.data_manipulation import convert_non_one_hot
    y_train = convert_non_one_hot(y_train)

    y_axis = []
    x_axis = [ 2 , 4 , 10 , 20 , 30 , 40 , 50 , 60];
    for iEpoch in x_axis :
        iAcc = gen_plot_res(X_train, y_train, X_valid, y_valid , iEpoch)
        y_axis.append(iAcc)
        if(iEpoch == 4):
            break;

    x_axis = x_axis[:len(y_axis)]
    fig = plt.figure( figsize=(11,8))
    ax1 = fig.add_subplot(111)
    ax1.plot( x_axis , y_axis , label ="Accuracy Graph" , color='c' , marker='o')


    plt.ylabel("Accuracy ")
    plt.xticks(x_axis)
    plt.xlabel("Epoch Count")

    handles , labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles , labels , loc = 'upper center' , bbox_to_anchor = (1.5 , 1) )
    ax1.grid('on')


    plt.savefig("")
    plt.show()


if __name__ == '__main__':
    main();