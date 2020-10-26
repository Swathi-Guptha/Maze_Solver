# Maze_Solver

# Description  
  
This project deals with solving a maze using a webcam. There was previous work done on perfect maze but our approach deals with solving all kinds of maze using a webcam. The only pre-requisite to solve the maze is, the video must be captured in well lit area so that the maze is clearly visible through webcam. It uses yoloV5 and Image processing to attain a solved maze. YoloV5 was trained of custom dataset.  
  
# Requirements  
To run the above code the requirements are:  
Setup:  
• Python 3.7  
• Laptop with a webcam 
• Pre-trained weights trained on yoloV5

  
To install all the modules required to run the code, you need to execte the below requirement:  
  
```python  
        pip install -r requirements.txt  
```  

# Run the project:  
1. Open cmd with the folder containing the project
2. The code to execte the detection file:  

```python  
        python yolov5/detect.py --weights last_yolov5s_results.pt --img 416 --conf 0.4 --source 0 --save-txt  
```
  
3. In order to smoothly execute the code a fluorescent light should be placed in front of the maze.  
4. To quit the program press 'q'.  

# Demo Video :  
• https://youtu.be/_pED1ANgbck

  
# Citation: 
• YoloV5 [https://github.com/ultralytics/yolov5#citation]  
  
# Contributors  
• @daggarwal01 [https://github.com/daggarwal01]

  


