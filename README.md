# SSHSNet for spine segmentation
Our model is built on linux, cuda10.1, python=3.6**， GeForce RTX 2080

For more information about SSHNet, please read the following paper:

    Meiyan Huang, Shuoling Zhou, Xiumei Chen, Haoran Lai, Qianjin Feng, Semi-supervised hybrid spine network for segmentation of spine MR images, Computerized Medical Imaging and Graphics, 2023, 107, 102245.          
      
Please also cite this paper if you are using SSHNet for your research!

##Package including:
* torch 1.7.1
* scikit-image 0.17.2
* scikit-learn 0.24.0
* SimpleITK 2.0.2
* nibabel 3.2.1
* nnunet 1.6.6
* numpy 1.19.4
* pandas 1.1.5
* argparse 1.4.0
* albumentations 0.5.2
* segmentation-models-pytorch 0.1.3
* tensorboard 2.4.1
* MedPy 0.4.0
* matplotlib 3.3.2

## Training steps

Commands for training：

    # Process 2D label data for training
    python process_data.py --filepath './train/MR' --maskpath "./train/Mask" --savepath "./dataset/processdata2D" --process2D True --withlabel True --infomation 'info.csv'
    
    # Process 2D unlabel data for training
    python process_data.py --filepath './test/MR' --savepath "./dataset/processdata2D" --process2D True --infomation 'unlabel_info.csv'
    
    # Process 3D label data for training
    python process_data.py --filepath './train/MR' --maskpath "./train/Mask" --savepath "./dataset/processdata3D" --withlabel True --infomation 'info.csv'
    
    # Split dataset
    The dataset has been splited, which are saved as 'splitdataset.pkl' and 'testdataset.pkl'
    
    # Train 2D network
    for fold in 0 1 2 3 4; do
        python train2d_semi_supervised.py --fold ${fold} --gpuid '0' --exid 'ex0' '--datapath' "./dataset/processdata2D" --train_batch_size 8 --seed 2021
    done
  
    # Train 3D network
    for fold in 0 1 2 3 4; do
        python train2D3D_concate.py --fold ${fold} --gpuid '0, 1' --exid 'ex1' --exid2D 'ex0' '--datapath' "./dataset/processdata3D" --seed 2021
    done
    
All weigthts will be saved in the file named 'weight/ex#/sub#'.
    
## Inference steps
Commands for evaluation of fivefold cross-validation:

    # Evaluate 2D network on validation set
    for fold in 0 1 2 3 4; do
        python inference2d.py --fold ${fold} --gpu '0' --ex 'ex0' --mainpath './dataset/process2Ddata/ --infomation 'info.csv' --standerpath '/train/Mask'
    done
    
    for fold in 0 1 2 3 4; do
        python evaluate.py --fold ${fold} --exid 'ex0' --standerpath './train/Mask'
    done
    
    # Evaluate SSHSNet on validation set 
    for fold in 0 1 2 3 4; do
        python inference3d.py --fold ${fold} --gpu '0, 1' --ex 'ex1' --mainpath './dataset/process3Ddata/ --infomation 'info.csv' --standerpath '/train/Mask'
    done
    
    for fold in 0 1 2 3 4; do
        python evaluate.py --fold ${fold} --exid 'ex1' --standerpath './train/Mask'
    done

Commands for prediction of testing set:
    
    # Process testing set
    python process_data.py --filepath './test/MR' --savepath "./dataset/processdata3D_test" --infomation 'info.csv'
    
    # Predict testing set
    python predict_fivefold.py --gpu '0,1' --exid2D 'ex0' --exid3D 'ex1' --datapath "./dataset/processdata3D_test" --oridatapath './test/MR' --batch_size 20 --infomation 'info.csv'
    
    The predicted results of testing set will be saved in the path "./dataset/processdata3D_test/ex1/predict"

    
    





