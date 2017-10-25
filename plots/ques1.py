from src.src_networks import large_main
import matplotlib.pyplot as plt

if __name__ == '__main__':

    for prefix in ["ques1b" , "ques1cb"]:

        if (prefix == 'ques1b'):
            internal = 'sigmoid'
            output = 'softmax'
            # ques1c means relu
        elif (prefix == 'ques1cb'):
            internal = 'relu'
            output = 'softmax'
        elif (prefix == 'ques1da'):
            # ques 1d means maxout
            exit(1);
        else:
            exit(1);


        y_axis = []
        x_axis_epoch = range(0 , 100 , 10)
        out = large_main(prefix).epoch_outputs

        for i in range(len(x_axis_epoch)):
            y_axis.append(out[x_axis_epoch[i]])


        x_axis = x_axis_epoch[:len(y_axis)]
        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_axis, y_axis, label="Accuracy Graph", color='c', marker='o')

        plt.ylabel("Accuracy ")
        plt.xticks(x_axis)
        plt.xlabel("Epoch Count")

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.5, 1))
        ax1.grid('on')

        plt.savefig("large_" + internal + "_"  + output)
        #plt.show()
        plt.clf()
