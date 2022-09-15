import time
import tkinter as tk
import random
import numpy as np
import math
import copy

def rgbToHex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

def drawCNN():
    global tCfgCNNSize, tCfgCNN, tCNN, tCObj
    #clear if there is existing drawing
    if len(tCObj)>0:
        for i in range(len(tCObj)):
            canvC.delete(tCObj[i])

    #Connections
    for k in range(len(tCfgCNNSize)-1):
        tW=tCNN["W" + str(k+1)]
        minValue=np.amin(tW)
        maxValue=np.amax(tW)
        tX1=(k+0.5)*tCWidth/(len(tCfgCNNSize))
        tX2=(k+1.5)*tCWidth/(len(tCfgCNNSize))
        for i in range(tCfgCNNSize[k]):
            tY1=(i+0.5)*tCHeight/(tCfgCNNSize[k])
            for j in range(tCfgCNNSize[k+1]):
                tY2=(j+0.5)*tCHeight/(tCfgCNNSize[k+1])
                if tW[j][i]>0:
                    colorValue=int(255*tW[j][i]/maxValue)
                    strColor=rgbToHex(255,255-colorValue,255-colorValue)
                elif tW[j][i]<0:
                    colorValue=abs(int(255*tW[j][i]/minValue))
                    strColor=rgbToHex(255-colorValue,255-colorValue,255)
                else:
                    strColor=rgbToHex(255,255,255)
                tO=canvC.create_line(tX1,tY1,tX2,tY2, width = 1, fill=strColor)
                tCObj.append(tO)
    #Draw nodes
    for i in range(len(tCfgCNNSize)):
        tX=(i+0.5)*tCWidth/(len(tCfgCNNSize))
        for j in range(tCfgCNNSize[i]):
            tY=(j+0.5)*tCHeight/(tCfgCNNSize[i])
            tO=canvC.create_oval(tX-2,tY-2,tX+2,tY+2, width = tLineWidth, outline='black')
            tCObj.append(tO)

def displayNetworkImage():
    global tCfgCNNSize, tCfgCNN, tCNN, tIObj
    #delete previous objects
    if len(tIObj)>0:
        for i in range(len(tIObj)):
            canvI.delete(tIObj[i])
    #Connections
    for k in range(len(tCfgCNNSize)-1):
        tW=tCNN["W" + str(k+1)]
        minValue=np.amin(tW)
        maxValue=np.amax(tW)
        #print(tW)
        tX1=(k+0.5)*tIWidth/(len(tCfgCNNSize))
        tX2=(k+1.5)*tIWidth/(len(tCfgCNNSize))
        dX=(tX2-tX1)*0.9
        for i in range(tCfgCNNSize[k]):
            tY1=0.05*tIHeight
            tY2=0.95*tIHeight
            dY=tY2-tY1
            for j in range(tCfgCNNSize[k+1]):
                if tW[j][i]>0:
                    colorValue=int(255*tW[j][i]/maxValue)
                    strColor=rgbToHex(255,255-colorValue,255-colorValue)
                elif tW[j][i]<0:
                    colorValue=abs(int(255*tW[j][i]/minValue))
                    strColor=rgbToHex(255-colorValue,255-colorValue,255)
                else:
                    strColor=rgbToHex(255,255,255)
                ttX1=tX1+i*dX/tCfgCNNSize[k]
                ttX2=tX1+(i+1)*dX/tCfgCNNSize[k]
                ttY1=tY1+j*dY/tCfgCNNSize[k+1]
                ttY2=tY1+(j+1)*dY/tCfgCNNSize[k+1]
                tO=canvI.create_rectangle(ttX1,ttY1,ttX2,ttY2,fill=strColor,outline='')
                tIObj.append(tO)

def drawAccuracy():
    global tCNNAccuracy, tAObj, tAHeight, tOffset, tAWidth
    #clear if there is existing drawing
    if len(tAObj)>0:
        for i in range(len(tAObj)):
            canvA.delete(tAObj[i])
    
    tOff=0.05*tAWidth

    #axis
    tO=canvA.create_line(tOff,tAHeight-tOff,tOff,tOff, width = 1, fill='black')
    tAObj.append(tO)
    tO=canvA.create_line(tOff,tAHeight-tOff,tAWidth-tOff,tAHeight-tOff, width = 1, fill='black')
    tAObj.append(tO)
    
    if len(tCNNAccuracy)==0:
        return
    if len(tCNNAccuracy)<100:
        for k in range(len(tCNNAccuracy)):
            tX=tOff+k*(tAWidth-tOff*2)/100
            tY=tOff+(1-tCNNAccuracy[k])*(tAHeight-tOff*2)
            tO=canvA.create_oval(tX-1,tY-1,tX+1,tY+1, width = 1, outline='red')
            tAObj.append(tO)            
    else:
        for k in range(len(tCNNAccuracy)):
            tX=tOff+k*(tAWidth-tOff*2)/len(tCNNAccuracy)
            tY=tOff+(1-tCNNAccuracy[k])*(tAHeight-tOff*2)
            tO=canvA.create_oval(tX-1,tY-1,tX+1,tY+1, width = 1, outline='red')
            tAObj.append(tO)            


def cCNNInit():
    global tCfgCNNSize, tCfgCNN, tCNN, tGuess, tCNNAccuracy, tAccumulatedEpoch
    inputString = inputNetworkArch.get(1.0, "end-1c")
    tCfgCNNSize=[9] #size of image
    tCNNAccuracy=[]
    tAccumulatedEpoch=0
    tGuess=[]
    
    ListString=inputString.split('\n')
    for i in range(len(ListString)):
        if len(ListString[i])>0:
            tCfgCNNSize.append(int(ListString[i]))
    tCfgCNNSize.append(9) #last output

    tCfgCNN=[]
    for i in range(len(tCfgCNNSize)-1):
        if i<len(tCfgCNNSize)-2:
            tCfgCNN.append({"input_dim": tCfgCNNSize[i], "output_dim": tCfgCNNSize[i+1], "activation": "relu"})
        else:
            tCfgCNN.append({"input_dim": tCfgCNNSize[i], "output_dim": tCfgCNNSize[i+1], "activation": "sigmoid"})

    #print(tCfgCNN)
    tCNN=init_layers(tCfgCNN, seed = 99)
    
    #display CNN
    #guessDigitPlot()
    drawCNN()
    displayNetworkImage()
    drawAccuracy()
    win.update()
            
   #CNN functions
def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
        
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        #Highlight nodes
        #drawForwardNode(idx,A_curr)
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    #Highlight last node
    #drawForwardNode(idx+1,A_curr)
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def find_winner(probs):
    winner = np.zeros(probs.shape) 
    winner[np.argmax(probs, axis=0), np.arange(probs.shape[1])] = 1
    #print(probs)
    #print(winner)
    return winner

#def get_accuracy_value(Y_hat, Y):
#    Y_hat_ = convert_prob_into_class(Y_hat)
#    return (Y_hat_ == Y).all(axis=0).mean()

def get_accuracy_value(Y_hat, Y):
    global tGuess
    Y_hat_ = find_winner(Y_hat)
    tGuess=(Y_hat_ == Y).all(axis=0)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    # iteration over network layers
    #for layer_idx, layer in enumerate(nn_architecture, 1):
    for layer_idx, layer in reversed(list(enumerate(nn_architecture, 1))):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
        #drawBackwardWeight(layer_idx, params_values["W" + str(layer_idx)])
    return params_values;

def train():
    global tNoEpoch, tLearnRate, tCNN, tCNNAccuracy, tStop, tAccumulatedEpoch, allBoard, allMove
    #numpy cnn code from https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
    
    # performing calculations for subsequent iterations
    for i in range(tNoEpoch):
        # step forward
        Y_hat, cashe = full_forward_propagation(np.transpose(np.array(allBoard)), tCNN, tCfgCNN)
        #print(Y_hat.shape)
        
        
        # calculating metrics and saving them in history
        #cost = get_cost_value(Y_hat, Y)
        #cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, np.transpose(np.array(allMove)))
        #print(accuracy)
        tCNNAccuracy.append(accuracy)
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, np.transpose(np.array(allMove)), cashe, tCNN, tCfgCNN)
        # updating model state
        tCNN = update(tCNN, grads_values, tCfgCNN, tLearnRate)
        #if stepDelay>0:
        #    time.sleep(stepDelay)
        if(i % 10 == 0):
            win.update()
            #print("Iteration: {:05} - accuracy: {:.5f}".format(i, accuracy))
            drawCNN()
            drawAccuracy()
            displayNetworkImage()
            #guessDigitPlot()
            win.update()

        if tStop==1:
            tStop=0
            textTrainingCount.config(text = "Epoch: %3d/%3d/%3d - stopped" % (i+1,tNoEpoch,tAccumulatedEpoch))
            return
        tAccumulatedEpoch+=1
        textTrainingCount.config(text = "Epoch: %3d/%3d/%3d - progress" % (i+1,tNoEpoch,tAccumulatedEpoch))
        textTrainingAccuracy.config(text = "Accuracy: {:.3f}".format(accuracy))            
    #return params_value 

#tic-tac-toe support functions
def checkWinner(board, player):
    #
    player=1
    for i in range(3):
        if board[i][0]==player and board[i][1]==player and board[i][2]==player:
            return player
        if board[0][i]==player and board[1][i]==player and board[2][i]==player:
            return player
    if board[0][0]==player and board[1][1]==player and board[2][2]==player:
        return player
    if board[0][2]==player and board[1][1]==player and board[2][0]==player:
        return player
    player=2
    for i in range(3):
        if board[i][0]==player and board[i][1]==player and board[i][2]==player:
            return player
        if board[0][i]==player and board[1][i]==player and board[2][i]==player:
            return player
    if board[0][0]==player and board[1][1]==player and board[2][2]==player:
        return player
    if board[0][2]==player and board[1][1]==player and board[2][0]==player:
        return player

    return 0

def updateScore():
    global tGameStat, lGameStat
    tD=tGameStat
    lGameStat.config(text = '/'.join(str(x) for x in tD))

def viewSelected():
    global tPlayOrder, tUserMark, tAI1Style, tAI2Style
    tPlayOrder = rbPlayOrder.get()
    tUserMark = rbUserMark.get()
    tAI1Style = rbAI1style.get()
    tAI2Style = rbAI2style.get()

def setMountOrigin(eventorigin):
    global mouseXY
    mouseXY=[eventorigin.x, eventorigin.y]
    #print(mouseXY)

def aiCount(tBoard):
    count=0
    for i in range(3):
        for j in range(3):
            if tBoard[i][j]>0:
                count+=1
    return count

def aiEmptyPairs(tBoard):
    emptyPairs=[]
    for i in range(3):
        for j in range(3):
            if tBoard[i][j]==0:
                emptyPairs.append([i, j])
    return emptyPairs

def aiEmptyCorner(tBoard):
    emptyPairs=[]
    for i in [0, 2]:
        for j in [0, 2]:
            if tBoard[i][j]==0:
                emptyPairs.append([i, j])
    return emptyPairs

def aiFindTwo(tBoard, player):
    #find two in a row cases
    #print(tBoard)
    #print('empty')
    emptyXY=aiEmptyPairs(tBoard)
    #print(emptyXY)
    placeXY=[]
    tempBoard=[]
    for k in range(len(emptyXY)):
        #print('check', emptyXY[k])
        tempBoard=copy.deepcopy(tBoard)
        tempBoard[emptyXY[k][0]][emptyXY[k][1]]=player
        #print('temp board', tempBoard)
        #check X
        #print('check X')
        count2=0
        count1=0
        for m in range(3):
            if tempBoard[m][emptyXY[k][1]]==player:
                count2+=1
            if tempBoard[m][emptyXY[k][1]]==3-player:
                count1+=1
        if count2==2 and count1==0:
            placeXY.append(emptyXY[k])
            #print(emptyXY[k])
        #check Y
        #print('check Y')
        count2=0
        count1=0
        for m in range(3):
            if tempBoard[emptyXY[k][0]][m]==player:
                count2+=1
            if tempBoard[emptyXY[k][0]][m]==3-player:
                count1+=1
        if count2==2 and count1==0:
            placeXY.append(emptyXY[k])
            #print(emptyXY[k])
        #check diagonal 1
        if emptyXY[k][0]==emptyXY[k][1]:
            count2=0
            count1=0
            #print('diagonal 1')
            for m in range(3):
                if tempBoard[m][m]==player:
                    count2+=1
                if tempBoard[m][m]==3-player:
                    count1+=1
            if count2==2 and count1==0:
                placeXY.append(emptyXY[k])
                #print(emptyXY[k])

        #check diagonal 2
        if emptyXY[k][0]+emptyXY[k][1]==2:
            #print('diagonal 2')
            count2=0
            count1=0
            for m in range(3):
                if tempBoard[m][2-m]==player:
                    count2+=1
                if tempBoard[m][2-m]==3-player:
                    count1+=1
            if count2==2 and count1==0:
                placeXY.append(emptyXY[k])
                #print(emptyXY[k])
    #print(placeXY)
    return placeXY

#algorithm players
def aiRandom(tBoard, player):
    x=np.random.randint(0,3)
    y=np.random.randint(0,3)
    while tBoard[x][y] > 0:
        x=np.random.randint(0,3)
        y=np.random.randint(0,3)
    tBoard[x][y]=player
    return [x,y]

def aiDefense(tBoard, player):
    if player==1:
        tLastMove=tP1LastMove
    else:
        tLastMove=tP2LastMove        
    count=aiCount(tBoard)
    #if first, random
    if count==0:
        x=np.random.randint(0,3)
        y=np.random.randint(0,3)
        while tBoard[x][y] > 0:
            x=np.random.randint(0,3)
            y=np.random.randint(0,3)
        tBoard[x][y]=player
        return [x,y]
    elif count==1:
        #place next to the first. Find empty around the tLastMove
        possibleXY=[]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                tX=i+tLastMove[0]
                tY=j+tLastMove[1]
                if tX>=0 and tX<=2 and tY>=0 and tY<=2:
                    if tBoard[tX][tY]==0:
                        possibleXY.append([tX,tY])
        #print(possibleXY)
        tI=np.random.randint(0,len(possibleXY))
        #print(tI)
        tBoard[possibleXY[tI][0]][possibleXY[tI][1]]=player
        return possibleXY[tI]
    else: #find out defense cases
        emptyXY=aiEmptyPairs(tBoard)
        #print(emptyXY)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(tBoard)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=3-player
            tempWin=checkWinner(tempBoard, 3-player)
            if tempWin>0:
                placeXY=emptyXY[k]
        #print(placeXY)
        if len(placeXY)>0:
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY
        else: #place next to opponent
            possibleXY=[]
            for i in [-1,1]:
                for j in [-1,1]:
                    tX=i+tLastMove[0]
                    tY=j+tLastMove[1]
                    if tX>=0 and tX<=2 and tY>=0 and tY<=2:
                        if tBoard[tX][tY]==0:
                            possibleXY.append([tX,tY])
            if len(possibleXY)>0:
                tI=np.random.randint(0,len(possibleXY))
                tBoard[possibleXY[tI][0]][possibleXY[tI][1]]=player
                return possibleXY[tI]
            else:
                tI=np.random.randint(0,len(emptyXY))
                tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
                return emptyXY[tI]

def aiOffense(tBoard, player):
    count=aiCount(tBoard)
    #if first, random
    if count==0: #place it at center
        tBoard[1][1]=player
        return [1,1]
    elif count==1:
        if tBoard[1][1]==0: #center if empty
            tBoard[1][1]=player
            return [1,1]
        else: #center was taken, place at corners
            emptyXY=aiEmptyCorner(tBoard)
            tI=np.random.randint(0,len(emptyXY))
            tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
            return emptyXY[tI]            
    elif count==2:
        #avoid ineffective directions and set preferred direction
        pXY=aiFindTwo(tBoard,player)
        tI=np.random.randint(0,len(pXY))
        tBoard[pXY[tI][0]][pXY[tI][1]]=player
        return pXY[tI]                    
    else:
        #find any winning positions
        emptyXY=aiEmptyPairs(tBoard)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(tBoard)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=player
            tempWin=checkWinner(tempBoard, player)
            if tempWin>0:
                placeXY=emptyXY[k]
        if len(placeXY)>0:
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY
        #if not, find positions for two in a row without opponent
        pXY=aiFindTwo(tBoard,player)
        if len(pXY)>0:
            tI=np.random.randint(0,len(pXY))
            tBoard[pXY[tI][0]][pXY[tI][1]]=player
            return pXY[tI]                    
        #if not random
        tI=np.random.randint(0,len(emptyXY))
        tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
        return emptyXY[tI]                    

def aiShallow(tBoard, player):
    count=aiCount(tBoard)
    #if first, random
    if count==0: #place it at center
        tBoard[1][1]=player
        return [1,1]
    elif count==1:
        if tBoard[1][1]==0: #center if empty
            tBoard[1][1]=player
            return [1,1]
        else: #center was taken, place at corners
            emptyXY=aiEmptyCorner(tBoard)
            tI=np.random.randint(0,len(emptyXY))
            tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
            return emptyXY[tI]            
    elif count==2:
        #avoid ineffective directions and set preferred direction
        pXY=aiFindTwo(tBoard,player)
        tI=np.random.randint(0,len(pXY))
        tBoard[pXY[tI][0]][pXY[tI][1]]=player
        return pXY[tI]                    
    else: #count>2
        #find any winning positions
        emptyXY=aiEmptyPairs(tBoard)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(tBoard)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=player
            tempWin=checkWinner(tempBoard, player)
            if tempWin>0:
                placeXY=emptyXY[k]
        if len(placeXY)>0:
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY
        #find any defensive positions
        emptyXY=aiEmptyPairs(tBoard)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(tBoard)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=3-player
            tempWin=checkWinner(tempBoard, 3-player)
            if tempWin>0:
                placeXY=emptyXY[k]
        if len(placeXY)>0:
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY

        #if not, find positions for two in a row without opponent
        pXY=aiFindTwo(tBoard,player)
        if len(pXY)>0:
            tI=np.random.randint(0,len(pXY))
            tBoard[pXY[tI][0]][pXY[tI][1]]=player
            return pXY[tI]                    
        #if not random
        tI=np.random.randint(0,len(emptyXY))
        tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
        return emptyXY[tI]                    

def aiSearch(tBoard, player):
    count=aiCount(tBoard)
    #if first, random
    if count==0: #place it at center
        tBoard[1][1]=player
        return [1,1]
    elif count==1:
        if tBoard[1][1]==0: #center if empty
            tBoard[1][1]=player
            return [1,1]
        else: #center was taken, place at corners
            emptyXY=aiEmptyCorner(tBoard)
            tI=np.random.randint(0,len(emptyXY))
            tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
            return emptyXY[tI]            
    elif count==2:
        #avoid ineffective directions and set preferred direction
        pXY=aiFindTwo(tBoard,player)
        tI=np.random.randint(0,len(pXY))
        tBoard[pXY[tI][0]][pXY[tI][1]]=player
        return pXY[tI]                    
    else: #count>2
        #find any winning positions
        emptyXY=aiEmptyPairs(tBoard)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(tBoard)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=player
            tempWin=checkWinner(tempBoard, player)
            if tempWin>0:
                placeXY=emptyXY[k]
        if len(placeXY)>0:
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY
        
        #find any defensive positions
        emptyXY=aiEmptyPairs(tBoard)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(tBoard)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=3-player
            tempWin=checkWinner(tempBoard, 3-player)
            if tempWin>0:
                placeXY=emptyXY[k]
        if len(placeXY)>0:
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY

        #find preemptive defensive positions
        [winPath1, winPath2]=searchNext(tBoard,player)
        if len(winPath1)>0 and player==2:
            tI=np.random.randint(0,len(winPath1))
            placeXY=winPath1[tI][1] #preemptive 2nd node place
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY
        if len(winPath2)>0 and player==1:
            tI=np.random.randint(0,len(winPath2))
            placeXY=winPath2[tI][1] #preemptive 2nd node place
            tBoard[placeXY[0]][placeXY[1]]=player
            return placeXY

        
        #if not, find positions for two in a row without opponent
        pXY=aiFindTwo(tBoard,player)
        if len(pXY)>0:
            tI=np.random.randint(0,len(pXY))
            tBoard[pXY[tI][0]][pXY[tI][1]]=player
            return pXY[tI]                    
        #if not random
        tI=np.random.randint(0,len(emptyXY))
        tBoard[emptyXY[tI][0]][emptyXY[tI][1]]=player
        return emptyXY[tI]                    

def searchNext(board, player):
    searchWeight=1
    emptyXY=aiEmptyPairs(board)
    #print(board)
    #winPath0=[]
    winPath1=[]
    winPath2=[]
    if len(emptyXY)==0: #draw
        return [winPath1, winPath2]

    #winning cases
    for k in range(len(emptyXY)):
        tempBoard=copy.deepcopy(board)
        tempBoard[emptyXY[k][0]][emptyXY[k][1]]=player
        tempWin=checkWinner(tempBoard, player)
        if tempWin>0:
            if tempWin==1:
                winPath1.append([emptyXY[k]])
            if tempWin==2:
                winPath2.append([emptyXY[k]])
    if len(winPath1)>0 or len(winPath2)>0:
        return [winPath1, winPath2]     

    else: #if there is no win, search for next step

        #find any defensive positions
        emptyXY=aiEmptyPairs(board)
        placeXY=[]
        tempBoard=[]
        for k in range(len(emptyXY)):
            tempBoard=copy.deepcopy(board)
            tempBoard[emptyXY[k][0]][emptyXY[k][1]]=3-player
            tempWin=checkWinner(tempBoard, 3-player)
            if tempWin>0:
                placeXY.append(emptyXY[k])
        if len(placeXY)>0:
            for k in range(len(placeXY)):
                tempBoard=copy.deepcopy(board)
                tempBoard[placeXY[k][0]][placeXY[k][1]]=player
                [tWinPath1, tWinPath2]=searchNext(tempBoard,3-player)
                for i in range(len(tWinPath1)):
                    if player==1:
                        tList=[placeXY[k]]
                    else:
                        tList=[]
                    for j in range(len(tWinPath1[i])):
                        tList.append(tWinPath1[i][j])
                    winPath1.append(tList)
                for i in range(len(tWinPath2)):
                    if player==1:
                        tList=[]
                    else:
                        tList=[placeXY[k]]
                    for j in range(len(tWinPath2[i])):
                        tList.append(tWinPath2[i][j])
                    winPath2.append(tList)
            return [winPath1, winPath2]    
        
        else: #if there is no defensive positions
            
            for k in range(len(emptyXY)):
                tempBoard=copy.deepcopy(board)
                tempBoard[emptyXY[k][0]][emptyXY[k][1]]=player
                [tWinPath1, tWinPath2]=searchNext(tempBoard,3-player)
                for i in range(len(tWinPath1)):
                    if player==1:
                        tList=[emptyXY[k]]
                    else:
                        tList=[]
                    for j in range(len(tWinPath1[i])):
                        tList.append(tWinPath1[i][j])
                    winPath1.append(tList)
                for i in range(len(tWinPath2)):
                    if player==1:
                        tList=[]
                    else:
                        tList=[emptyXY[k]]
                    for j in range(len(tWinPath2[i])):
                        tList.append(tWinPath2[i][j])
                    winPath2.append(tList)
            #if len(winPath1)==1:
            return [winPath1, winPath2]

def aiMachine(tBoard, player):
    global tCNN, tCfgCNN
    tData=np.array(convertBoard(tBoard, player))
    tVec=np.reshape(tData,[1,9])
    #tVec=np.reshape(np.array(convertBoard(tBoard, player)),[1,9])
    #print('tData: ', tData)
    #print('tVec: ', tVec)
    #print(tVec.shape)
    Y_out, cashe = full_forward_propagation(np.transpose(tVec), tCNN, tCfgCNN)
    #Y_hat=Y_out.flatten()
    #tidx=np.argsort(Y_hat)
    #idx=np.flip(tidx)
    idx=np.flip(np.argsort(Y_out.flatten()))
    #print('Y_hat: ', Y_hat)
    #print('idx: ', idx)
    xy=[]
    i=0
    while True:
        if tData[idx[i]]==0:
            tX=idx[i]//3
            tY=idx[i]%3
            #print("tX,tX: ", tX,tY)
            tBoard[tX][tY]=player
            return [tX, tY]
        i+=1
    #find out maximum out of available spots
    return

#GUI support functions and main GUI function
def cNoSample():
    global tNoGames
    inp = inputText.get(1.0, "end-1c")
    tNoGames = int(inp)

def cAnimFaster():
    global stepDelay
    if stepDelay>0:
        stepDelay=round(10*(stepDelay-0.1))/10
        inputSpeed.delete('1.0', tk.END)
        inputSpeed.insert(tk.END, stepDelay)
        
def cAnimSlower():
    global stepDelay
    stepDelay=round(10*(stepDelay+0.1))/10
    inputSpeed.delete('1.0', tk.END)
    inputSpeed.insert(tk.END, stepDelay)

def cAnimationDelay():
    global stepDelay
    inp = inputSpeed.get(1.0, "end-1c")
    stepDelay = float(inp)

def cTrain():
    global tNoEpoch, tLearnRate
    inp = inputNoEpoch.get(1.0, "end-1c")
    tNoEpoch = int(inp)
    inp = inputLearnRate.get(1.0, "end-1c")
    tLearnRate = float(inp)

    if len(tCfgCNN)==0:
        cCNNInit()
        
    train()

def convertBoard(board, player):
    array=sum(board,[])
    for i in range(len(array)):
        if array[i]==player:
            array[i]=1
        elif array[i]==3-player:
            array[i]=-1
        else:
            array[i]=0
    return array

def convertPlay(xy):
    move=[0]*9
    idx=xy[0]*3+xy[1]
    move[idx]=1
    return move

def cPlayMatch():
    global tObj, mouseXY, tCoord, tGameStat, tLastMove, tBoard, tNoGames, tP1LastMove, tP2LastMove
    global allBoardP1, allBoardP2, allMoveP1, allMoveP2, allBoard, allMove

    inp = inputNoGame.get(1.0, "end-1c")
    tNoGames = int(inp)
    
    tGameStat=[0,0,0,0]
    tPlayTurn=0
    allBoard=[]
    allMove=[]
    
    for g in range(tNoGames):
        #start match
        #print(len(allBoard))
        allBoardP1=[]
        allBoardP2=[]
        allMoveP1=[]
        allMoveP2=[]

        mBoard=[]
        tBoard=[[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        lGameOver.config(text="")
        #1 for player, and 2 for AI always
        if len(tObj)>0:
            for i in range(len(tObj)):
                canv.delete(tObj[i])
        tObj=[]
        tCount=0
        tWin=0
        mouseXY=[0,0]
        win.update()
        #Set starting player
        if tPlayOrder == 1:
            tPlayTurn=1
        elif tPlayOrder == 2:
            tPlayTurn=2
        elif tPlayOrder == 3:
            tPlayTurn=(g % 2) + 1
        while tCount<9 and tWin==0:
            #player order
            if tPlayTurn == 1:
                xy=aiMachine(tBoard, 1)
                #store play data
                allBoardP1.append(convertBoard(tBoard, 1))
                allMoveP1.append(convertPlay(xy))
                #display AI mark
                tX=tCoord[xy[0]]
                tY=tCoord[xy[1]]
                if tUserMark==1:
                    tL1=canv.create_line(tX-30,tY-30,tX+30,tY+30, width = tLineWidth, fill='green')
                    tL2=canv.create_line(tX+30,tY-30,tX-30,tY+30, width = tLineWidth, fill='green')
                    tObj.append(tL1)
                    tObj.append(tL2)
                else:
                    tL1=canv.create_oval(tX-30,tY-30,tX+30,tY+30, width = tLineWidth, outline='green')
                    tObj.append(tL1)
                #tWin=checkWinner(tBoard, 1)
                tP1LastMove = xy
                tPlayTurn = 2
                mBoard.append(copy.deepcopy(tBoard))
            else: #call AI player
                if tAI2Style==1:
                    #random player
                    xy=aiRandom(tBoard, 2)                
                elif tAI2Style==2:
                    #defense player
                    xy=aiDefense(tBoard, 2)
                elif tAI2Style==3:
                    #offense player
                    xy=aiOffense(tBoard, 2)
                elif tAI2Style==4:
                    #shallow search
                    xy=aiShallow(tBoard, 2)
                elif tAI2Style==5:
                    #complete search
                    xy=aiSearch(tBoard, 2)
                elif tAI2Style==6:
                    #for machine vs. machine learning
                    xy=aiMachine(tBoard, 2)

                #store play data
                allBoardP2.append(convertBoard(tBoard, 2))
                allMoveP2.append(convertPlay(xy))
                #display AI mark
                tX=tCoord[xy[0]]
                tY=tCoord[xy[1]]
                if tUserMark==2:
                    tL1=canv.create_line(tX-30,tY-30,tX+30,tY+30, width = tLineWidth, fill='orange')
                    tL2=canv.create_line(tX+30,tY-30,tX-30,tY+30, width = tLineWidth, fill='orange')
                    tObj.append(tL1)
                    tObj.append(tL2)
                else:
                    tL1=canv.create_oval(tX-30,tY-30,tX+30,tY+30, width = tLineWidth, outline='orange')
                    tObj.append(tL1)
                #check winner for player 2
                #print('AI placed:',tBoard)
                
                #print('AI wins:',tWin)
                tP2LastMove = xy
                tPlayTurn = 1
                mBoard.append(copy.deepcopy(tBoard))
            if stepDelay>0:
                time.sleep(stepDelay)
            win.update()
            tWin=checkWinner(tBoard, 1)
            tCount+=1
            #print(tBoard)
        #announce results - win or draw    
        tGameStat[3]+=1
        if tWin==0:
            lGameOver.config(text="Draw")
            tGameStat[1]+=1
        elif tWin==1:
            lGameOver.config(text="Toothless won!")
            tGameStat[0]+=1
            allBoard=allBoard+allBoardP1
            allMove=allMove+allMoveP1

        elif tWin==2:
            lGameOver.config(text="Pouncer won!")
            tGameStat[2]+=1
            #for k in range(len(mBoard)):
                #printBoard(mBoard[k])
            allBoard=allBoard+allBoardP2
            allMove=allMove+allMoveP2

        updateScore()
        win.update()
        #if stepDelay>0:
        #    time.sleep(stepDelay*2)

#main variables
xWinSize=1024
yWinSize=768
noSample=100
stepDelay=0
winFontSize=20
listFontSize=12
yLineUp=0.5
yCompare=0.6

tSize=270
tOffset=3
tLineWidth=5

tPlayOrder=3
tUserMark=1
tAI1Style=1
tAI2Style=1
tNoGames=100

tStrCfgCNNSize='10\n10'
tCfgCNNSize=[10,10]
tCfgCNN=[]
tNoEpoch=100
tLearnRate=0.01
tStop=0
tAccumulatedEpoch=0

tCWidth=600
tCHeight=270
tCXOffset=5
tCYOffset=5
tCNNAccuracy=[]
tLearnRate=0.01
tCNN=[]
tCNNAccuracy=[]
tCObj=[]

tIWidth=600
tIHeight=270
tIObj=[]

tAWidth=270
tAHeight=270
tAObj=[]

tP1LastMove=[]
tP2LastMove=[]
tObj=[]
tGameStat=[0,0,0,0]
mouseXY=[0,0]

allBoard=[]
allBoardP1=[]
allBoardP2=[]
allMove=[]
allMoveP1=[]
allMoveP2=[]

#noArray=[]
#widgetList=[]
#noList=[]
#dispListIn=[]
#dispListOut=[]

#Create an instance of tkinter frame
win=tk.Tk()
win.title("Tic-Tac-Toe AI Machine Learning")
rbAI1style = tk.IntVar()
rbAI1style.set(tAI1Style)
rbAI2style = tk.IntVar()
rbAI2style.set(tAI2Style)
rbUserMark = tk.IntVar()
rbUserMark.set(tUserMark)
rbPlayOrder = tk.IntVar()
rbPlayOrder.set(tPlayOrder)
#Define the geometry of window
win.geometry("1024x768")

xP1=(tSize-tOffset*2)/3+tOffset
xP2=(tSize-tOffset*2)/3*2+tOffset
tCoord=[0.5*(tSize-tOffset*2)/3+tOffset, 1.5*(tSize-tOffset*2)/3+tOffset, 2.5*(tSize-tOffset*2)/3+tOffset]


#tic-tac-toe board
canv=tk.Canvas(win, width=tSize, height=tSize)
canv.place(relx=0.70, rely=0.29, anchor=tk.NW)
canv.create_line(xP1,tOffset,xP1,tSize-tOffset, width = tLineWidth, fill='black')
canv.create_line(xP2,tOffset,xP2,tSize-tOffset, width = tLineWidth, fill='black')
canv.create_line(tOffset,xP1,tSize-tOffset,xP1, width = tLineWidth, fill='black')
canv.create_line(tOffset,xP2,tSize-tOffset,xP2, width = tLineWidth, fill='black')
tttRect1=canv.create_rectangle(tOffset,tOffset,tSize-tOffset,tSize-tOffset,outline="gray")
#limit mouse click to canvas
canv.bind("<Button 1>",setMountOrigin)

#CNN display
canvC=tk.Canvas(win, width=tCWidth, height=tCHeight)
canvC.place(relx=0.05, rely=0.29, anchor=tk.NW)
tttRect2=canvC.create_rectangle(tOffset,tOffset,tCWidth-tOffset,tCHeight-tOffset,outline="gray")

#CNN matrix image display
canvI=tk.Canvas(win, width=tIWidth, height=tIHeight)
canvI.place(relx=0.05, rely=0.64, anchor=tk.NW)
tttRectI=canvI.create_rectangle(tOffset,tOffset,tIWidth-tOffset,tIHeight-tOffset,outline="gray")

#CNN accuracy display
canvA=tk.Canvas(win, width=tAWidth, height=tAHeight)
canvA.place(relx=0.70, rely=0.64, anchor=tk.NW)
tttRectA=canvA.create_rectangle(tOffset,tOffset,tAWidth-tOffset,tAHeight-tOffset,outline="gray")


#left column
textNetworkArch = tk.Label(win, text = "# of Nodes at Layer", font=("Arial", winFontSize))
textNetworkArch.place(relx=0.01, rely=0.02, anchor=tk.W)

inputNetworkArch = tk.Text(win, height = 7, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputNetworkArch.place(relx=0.01, rely=0.04, anchor=tk.NW)
inputNetworkArch.insert(tk.END, tStrCfgCNNSize)

bNetworkInitialize = tk.Button(win, text = "Initialize",  command = cCNNInit, font=("Arial", winFontSize))
bNetworkInitialize.place(relx=0.08, rely=0.06, anchor=tk.W)

#Center column
#play order radio button
lPlayOrder = tk.Label(win, text = "Alternating order", font=("Arial", winFontSize))
lPlayOrder.place(relx=0.08, rely=0.10, anchor=tk.W)
#bPlayOrderUser=tk.Radiobutton(win, text="Toothless first", variable=rbPlayOrder, value=1, command=viewSelected, font=("Arial", winFontSize))
#bPlayOrderUser.place(relx=0.27, rely=0.02, anchor=tk.W)
#bPlayOrderAI=tk.Radiobutton(win, text="Pouncer first", variable=rbPlayOrder, value=2, command=viewSelected, font=("Arial", winFontSize))
#bPlayOrderAI.place(relx=0.27, rely=0.06, anchor=tk.W)
#bPlayOrderAI=tk.Radiobutton(win, text="Alternating", variable=rbPlayOrder, value=3, command=viewSelected, font=("Arial", winFontSize))
#bPlayOrderAI.place(relx=0.27, rely=0.10, anchor=tk.W)

    
#player shape radio button
lPlayMark= tk.Label(win, text = "Toothless uses X", font=("Arial", winFontSize))
lPlayMark.place(relx=0.08, rely=0.14, anchor=tk.W)
#bStyleX=tk.Radiobutton(win, text="Toothless uses X", variable=rbUserMark, value=1, command=viewSelected, font=("Arial", winFontSize))
#bStyleX.place(relx=0.27, rely=0.14, anchor=tk.W)
#bStyleO=tk.Radiobutton(win, text="Toothless uses O", variable=rbUserMark, value=2, command=viewSelected, font=("Arial", winFontSize))
#bStyleO.place(relx=0.27, rely=0.18, anchor=tk.W)

#4th column
textNoGame = tk.Label(win, text = "# of games", font=("Arial", winFontSize))
textNoGame.place(relx=0.55, rely=0.02, anchor=tk.W)

inputNoGame = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputNoGame.place(relx=0.70, rely=0.02, anchor=tk.W)
inputNoGame.insert(tk.END, tNoGames)

#
textNoEpoch = tk.Label(win, text = "# of epochs", font=("Arial", winFontSize))
textNoEpoch.place(relx=0.55, rely=0.06, anchor=tk.W)

inputNoEpoch = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputNoEpoch.place(relx=0.70, rely=0.06, anchor=tk.W)
inputNoEpoch.insert(tk.END, tNoEpoch)

#
textLearnRate = tk.Label(win, text = "Learning rate", font=("Arial", winFontSize))
textLearnRate.place(relx=0.55, rely=0.10, anchor=tk.W)

inputLearnRate= tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputLearnRate.place(relx=0.70, rely=0.10, anchor=tk.W)
inputLearnRate.insert(tk.END, tLearnRate)

#
#textSpeed = tk.Label(win, text = "Delay (sec)", font=("Arial", winFontSize))
#textSpeed.place(relx=0.70, rely=0.14, anchor=tk.W)

#inputSpeed = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
#inputSpeed.place(relx=0.85, rely=0.14, anchor=tk.W)
#inputSpeed.insert(tk.END, stepDelay)

#bFaster = tk.Button(win, text = "Faster",  command = cAnimFaster, font=("Arial", winFontSize))
#bFaster.place(relx=0.70, rely=0.18, anchor=tk.W)

#bSlower = tk.Button(win, text = "Slower",  command = cAnimSlower, font=("Arial", winFontSize))
#bSlower.place(relx=0.85, rely=0.18, anchor=tk.W)

#start match button
bPlay = tk.Button(win, text = "Play games",  command = cPlayMatch, font=("Arial", winFontSize))
bPlay.place(relx=0.55, rely=0.14, anchor=tk.W)

#start train button
bTrain = tk.Button(win, text = "Start training",  command = cTrain, font=("Arial", winFontSize))
bTrain.place(relx=0.70, rely=0.14, anchor=tk.W)


#Right column
#AI player type selection
lAIType= tk.Label(win, text = "Trainer AI player style", font=("Arial", winFontSize))
lAIType.place(relx=0.30, rely=0.02, anchor=tk.W)

b2Random=tk.Radiobutton(win, text="Random", variable=rbAI2style, value=1, command=viewSelected, font=("Arial", winFontSize))
b2Random.place(relx=0.30, rely=0.06, anchor=tk.W)
b2Defense=tk.Radiobutton(win, text="Defensive", variable=rbAI2style, value=2, command=viewSelected, font=("Arial", winFontSize))
b2Defense.place(relx=0.30, rely=0.10, anchor=tk.W)
b2Attack=tk.Radiobutton(win, text="Offensive", variable=rbAI2style, value=3, command=viewSelected, font=("Arial", winFontSize))
b2Attack.place(relx=0.30, rely=0.14, anchor=tk.W)
b2SearchShallow=tk.Radiobutton(win, text="Defense and offense", variable=rbAI2style, value=4, command=viewSelected, font=("Arial", winFontSize))
b2SearchShallow.place(relx=0.30, rely=0.18, anchor=tk.W)
b2SearchComplete=tk.Radiobutton(win, text="Search", variable=rbAI2style, value=5, command=viewSelected, font=("Arial", winFontSize))
b2SearchComplete.place(relx=0.30, rely=0.22, anchor=tk.W)
b2SearchComplete=tk.Radiobutton(win, text="Machine learning", variable=rbAI2style, value=6, command=viewSelected, font=("Arial", winFontSize))
b2SearchComplete.place(relx=0.30, rely=0.26, anchor=tk.W)


lGameOver= tk.Label(win, text = "", font=("Arial", winFontSize))
lGameOver.place(relx=0.55, rely=0.18, anchor=tk.W)

lGameCount=tk.Label(win, text = "W/D/L/Total", font=("Arial", winFontSize))
lGameCount.place(relx=0.55, rely=0.22, anchor=tk.W)

lGameStat=tk.Label(win, text = '/'.join(str(x) for x in tGameStat), font=("Arial", winFontSize))
lGameStat.place(relx=0.55, rely=0.26, anchor=tk.W)

textTrainingCount = tk.Label(win, text = "Epoch:", font=("Arial", winFontSize))
textTrainingCount.place(relx=0.70, rely=0.18, anchor=tk.W)
textTrainingAccuracy = tk.Label(win, text = "Accuracy:", font=("Arial", winFontSize))
textTrainingAccuracy.place(relx=0.70, rely=0.22, anchor=tk.W)

win.mainloop()