import numpy as np

from src.src_networks import small_main
import matplotlib.pyplot as plt
MAX_EPOCH  = 51

if __name__ == '__main__':

    for prefix in ["ques1a" , "ques1ca" , 'ques1da' ]:

        if (prefix == 'ques1a'):
            internal = 'sigmoid'
            output = 'sigmoid'
            # ques1c means relu
        elif (prefix == 'ques1ca'):
            internal = 'relu'
            output = 'sigmoid'
        elif (prefix == 'ques1da'):
            internal = 'maxout'
            output = 'sigmoid'
        else:
            exit(1);


        x_axis_epoch = range(0 , MAX_EPOCH , 2)
        all_models = small_main(internal , output)

        y_stack = None

        for i in range(len(all_models)):
            y_axis = []
            out = all_models[i].epoch_outputs

            print("out Shape" , len(out)  , np.shape(out))

            for j in range(len(x_axis_epoch)):
                y_axis.append( out[x_axis_epoch[j]])

            y_axis = np.asarray( y_axis );
            if y_stack is not None:
                y_stack  = np.row_stack((y_stack  , y_axis));
            else:
                y_stack = np.asarray(y_axis)

        x_axis = x_axis_epoch[:len(y_axis)]
        y_axis = None;

        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(111)

        ax1.plot(x_axis, y_stack[0, :], label='Fold 1', color='c', marker='o')
        ax1.plot(x_axis, y_stack[1, :], label='Fold 2', color='g', marker='o')
        ax1.plot(x_axis, y_stack[2, :], label='Fold 3', color='r', marker='o')
        ax1.plot(x_axis, y_stack[3, :], label='Best Model', color='b', marker='o')

        plt.ylabel("Accuracy ")
        plt.xticks(x_axis)
        plt.xlabel("Epoch Count")

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels)
        ax1.grid('on')

        plt.savefig("large_" + internal + "_"  + output)
        #plt.show()
        plt.clf()
