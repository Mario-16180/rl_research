import wandb
import random
import numpy as np

if __name__ == '__main__':
    # start a new wandb run to track this script
    run1 = wandb.init(
        # set the wandb project where this run will be logged
        project="explore_wandb",
        #id="valiant-sunset-44"
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 100,
        },
        resume=False
    )

    run1.define_metric("epoch")
    run1.define_metric("train/*", step_metric="epoch")
    # simulate training
    epochs = 100
    offset = random.random() / 5
    eval_reward_matrix = []
    for epoch in range(epochs+1):
        reward_train = 0.8 - 2 ** -(epoch+2) - random.random() / (epoch+2) - offset
        loss_train = 2 ** -(epoch+2) + random.random() / (epoch+2) + offset
        # log metrics to wandb
        run1.log({"epoch": epoch, "train/reward": reward_train, "train/loss": loss_train})
        if epoch % 10 == 0:
            eval_reward_episode = []
            for i in range(20):
                reward_test = 0.8 - 2 ** -(epoch+2) - random.random() / (epoch+2) - offset
                run1.log({f"train/eval_{i}": reward_test})
                eval_reward_episode.append(reward_test)
            eval_reward_matrix.append(eval_reward_episode)
            # log metrics to wandb

    # transpose list of list eval_reward_matrix
    #eval_reward_matrix = np.transpose(eval_reward_matrix)
    #for i in range(len(eval_reward_matrix)):
    #    for k in range(len(eval_reward_matrix[i])):
    #        run1.log({"epoch": k*10, f"train/eval_{i}": eval_reward_matrix[i][k]})

    # [optional] finish the wandb run, necessary in notebooks
    run1.save
    run1.finish()