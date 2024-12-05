# Motion Prediction

This guide explains how to set up and run motion prediction using the TAROT framework and the [UniTraj](https://github.com/vita-epfl/UniTraj) repository.

## Step-by-Step Instructions

1. **Navigate to the motion prediction example directory**  
   `cd examples/motion_prediction`

2. **Clone the UniTraj repository**  
   Ensure you have SSH access to GitHub before running the following command:  
   `git clone git@github.com:vita-epfl/UniTraj.git`

3. **Install necessary dependencies for UniTraj**  
   Follow the instructions in the [UniTraj README](./UniTraj/README.md) to set up dependencies. 

4. **Move TAROT-specific scripts into the UniTraj directory**  
   Move the `tarot_train.py` and `tarot_config.yaml` files into the cloned UniTraj repository:  

   `mv tarot_train.py UniTraj/unitraj`

   `mv tarot_config.yaml UniTraj/unitraj`

   Download pretrained Autobot [Ckpt](https://drive.google.com/file/d/19Ak1Is2JEzI8QNusV2j2Z2i3cFzxyCVP/view?usp=sharing) into UniTraj/unitraj/unitraj_ckpt



5. **Navigate to the UniTraj directory**  
   `cd UniTraj`

6. **Run the TAROT training script**  
   Use the following command to start the training process:  
   `python tarot_train.py`

## Notes

- Make sure to configure `tarot_config.yaml` according to your requirements before running the training script.
- For issues with dependencies or environment setup, refer to the [UniTraj documentation](https://github.com/vita-epfl/UniTraj) or TAROT-specific documentation.

## Example Directory Structure

After completing the steps, your directory structure should look like this:

examples/motion_prediction/  
├── UniTraj/unitraj  
│   ├── tarot_train.py  
│   ├── tarot_config.yaml  
│   └── ... (other UniTraj files)  