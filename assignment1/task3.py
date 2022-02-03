import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    batch_size = np.size(X, 0)
    outputs = model.forward(X)
    correct = 0
    for image in range(batch_size):
        pred_number = np.argmax(outputs[image])
        label = np.argmax(targets[image])
        if pred_number == label:
            correct += 1
    # TODO: Implement this function (task 3c)
    accuracy = correct / batch_size
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        targets = Y_batch
        outputs = self.model.forward(X_batch)
        
        loss = cross_entropy_loss(targets, outputs)
        self.model.backward(X_batch, outputs, targets)
        self.model.w = self.model.w - self.learning_rate * self.model.grad
        # TODO: Implement this function (task 3b)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    # model = SoftmaxModel(l2_reg_lambda=0)
    # # Train model
    # trainer = SoftmaxTrainer(
    #     model, learning_rate, batch_size, shuffle_dataset,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history, val_history = trainer.train(num_epochs)

    # print("Final Train Cross Entropy Loss:",
    #       cross_entropy_loss(Y_train, model.forward(X_train)))
    # print("Final Validation Cross Entropy Loss:",
    #       cross_entropy_loss(Y_val, model.forward(X_val)))
    # print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    # print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # plt.ylim([0.2, .6])
    # utils.plot_loss(train_history["loss"],
    #                 "Training Loss", npoints_to_average=10)
    # utils.plot_loss(val_history["loss"], "Validation Loss")
    # plt.legend()
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Cross Entropy Loss - Average")
    # plt.savefig("task3b_softmax_train_loss.png")
    # plt.show()

    # # Plot accuracy
    # plt.ylim([0.89, .93])
    # utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    # utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig("task3b_softmax_train_accuracy.png")
    # plt.show()

    # # Train a model with L2 regularization (task 4b)

    # model1 = SoftmaxModel(l2_reg_lambda=2.0)
    # trainer = SoftmaxTrainer(
    #     model1, learning_rate, batch_size, shuffle_dataset,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # # You can finish the rest of task 4 below this point.

    # print("Final Train Cross Entropy Loss:",
    #       cross_entropy_loss(Y_train, model.forward(X_train)))
    # print("Final Validation Cross Entropy Loss:",
    #       cross_entropy_loss(Y_val, model.forward(X_val)))
    # print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model1))
    # print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model1))

    # #plt.ylim([0.2, .6])
    # utils.plot_loss(train_history_reg01["loss"],
    #                 "Training Loss", npoints_to_average=10)
    # utils.plot_loss(val_history_reg01["loss"], "Validation Loss")
    # plt.legend()
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Cross Entropy Loss - Average")
    # plt.savefig("task4b_softmax_train_loss.png")
    # plt.show()

    # # Plot accuracy
    # #plt.ylim([0.89, .93])
    # utils.plot_loss(train_history_reg01["accuracy"], "Training Accuracy")
    # utils.plot_loss(val_history_reg01["accuracy"], "Validation Accuracy")
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig("task4b_softmax_train_accuracy.png")
    # plt.show()

    # # Plotting of softmax weights (Task 4b)
    # fig, axis = plt.subplots(4,5)
    # weights = np.hstack((model.w,model1.w)).T
    # for i,sub_plot in enumerate(axis.ravel()):
    #     fig_row = i//5
    #     fig_column = i%5
    #     w_img = np.reshape(weights[i,:-1], (28,28))
    #     sub_plot.imshow(w_img, cmap="gray")
    # plt.savefig("task4b_softmax_weight.png")
    # plt.show()
    

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [2, .2, .02, .002]
    val_accs = []
    weights_lam = []
    for i, lam in enumerate(l2_lambdas):
        model_lambda = SoftmaxModel(l2_reg_lambda=lam)
        trainer = SoftmaxTrainer(
            model_lambda, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg_lam, val_history_reg_lam = trainer.train(num_epochs)
        weights_lam.append(model_lambda.w)
        val_accs.append(val_history_reg_lam["accuracy"])

    # Plot accuracy
    for i,acc in enumerate(val_accs):
        x = [*acc.keys()]
        y = [acc[key] for key in x]
        plt.plot(x,y)
             
    plt.ylim([0.75,0.95])    
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend(["lambda=" + str(lam) for lam in l2_lambdas])
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    l2_norms = [w.ravel().dot(w.ravel()) for w in weights_lam]
    plt.plot(l2_lambdas, l2_norms)
    plt.xlabel("Lambda")
    plt.ylabel("L2-norm of weights")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()
