"""DQN_MODIFIED_VERSION_17 AUG 23"""
import os
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Tartarus import Tarpy
from controller import Robot


TIME_STEP = 100 #in millisconds #increase / decrease for slowing speeding up the simulation
MAX_SPEED = 6.28
send_weights_iter=100 #change this to change the num of iterations afgter which the weights are sent


robot=Robot()

#initialize devices
ps=[0,0,0,0,0]
psNames=["ps0", "ps2", "ps4","ps5","ps7"] #two front ones 7,0; two side ones 5,2; one back 4
ps=[robot.getDevice(name) for name in psNames]
sampling_period=50 #in ms  It determines how often the proximity sensors will be sampled to get their values. A smaller value means the sensors will be read more frequently, while a larger value reduces the frequency.
for sensor in ps:
    sensor.enable(sampling_period)

# Initialize motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
psValues = [0,0,0,0,0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

  



# Define the DQN neural network architecture
#tweak nn architecture #readmore on the layers used-- pertaining to the standard implementation?
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#Function to read sensor values from the robot
def read_sensors():
    psValues = []
    psValues=[sensor.getValue() for sensor in ps]
    #proximity_values_meter = [value * sensor.getLookupTable()[0] for value, sensor in zip(psValues, ps)]
    #for i in psValues:
       #     print("psvals  ", i)
    return psValues

# # Function to convert sensor values to binary representation (obstacle or not)
# def process_sensors(sensor_values):
#     threshold = 75  # threshold is 75 because sensor values are around 60-70 if theres no obstacle. sensor values are around 100-500 if there are obstacles infront
#     binary_sensors = [1 if sensor_value > threshold else 0 for sensor_value in sensor_values]
#     return binary_sensors

# # Function to normalize sensor values #helps when performing real robot experiments #values are between 0 and 1
# #34 and 4095 are the minimum and maximum values of the proximity sensor in the webots epuck
# def normalize_sensor_values(sensor_values):
#     min_value = 50    #34
#     max_value = 5000      #4095
#     normalized_values = [(value - min_value) / (max_value - min_value) for value in sensor_values]
#     #normalized_values=sensor_values
#     return normalized_values

#FUNCTION TO CONVERT THE CONTINUOUS SENSOR VALUES INTO BINS
def bin_sensor_values(sensor_values):
    no_obstacle_thresh=75
    obstacle_proximity_thresh=175
    obstacle_close_thresh=1000
    bin_sensors=[]
    for val in sensor_values:
        #code for if val < 75 then  val=0 elif 75<=val<175 then val=1 elif 175<=val<1000 then val=2 elif val>1000 then val=3
        if val<no_obstacle_thresh:
            val=0
        elif val>=no_obstacle_thresh and val<obstacle_proximity_thresh:
            val=1
        elif val>=obstacle_proximity_thresh and val<obstacle_close_thresh:
            val=2
        elif val>=obstacle_close_thresh:
            val=3
        bin_sensors.append(val)

    return bin_sensors

# Define the path to the log file
log_file_path = "batch_logs.txt"

  

# Function to select an action using epsilon-greedy policy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
      #  print("random action")
        return np.random.choice(output_size), 0
    else:
        with torch.no_grad(): #disable gradient calculation #only forward prop
     #       print("NN Based Action")
            return model(state).argmax().item(), model(state).max().item()#return the index of the maximum value in the tensor #returnmax q value also added newly
        #note down the q value that this returns

# to control the robot's movement
def move_forward():
    leftSpeed = 0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("forward")
    time.sleep(0.1)

def backward():
    leftSpeed = -0.2 * MAX_SPEED
    rightSpeed = -0.2 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("backward")
    time.sleep(0.1)


def left():
    leftSpeed = -0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("left")
    time.sleep(0.1)

def right():
    leftSpeed = 0.5 * MAX_SPEED
    rightSpeed = -0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    #print("right")
    time.sleep(0.1)

# def stop():
#     leftSpeed = 0
#     rightSpeed = 0
#     leftMotor.setVelocity(leftSpeed)
#     rightMotor.setVelocity(rightSpeed)
#   ##  print("stop")
#     time.sleep(0.1)

def calculate_reward(previous_state, current_state, action):
    reward=0
    # Loop through each sensor and check for transitions
    for prev_val, curr_val in zip(previous_state, current_state):
        #0->0 +10; 0->1 +1; 0->2 -5; 0->3 -10;
        #1->0 +5; 1->1 +1; 1->2 -5; 1->3 -10;
        #2->0 +10; 2->1 +5; 2->2 -5; 2->3 -10;
        if prev_val==0 and curr_val==0 and all(curr==0 for curr in current_state)==0:
            reward+=2
            #print("1. +10")
        elif prev_val==0 and curr_val==1:
            reward+=0.1
            #print("2. +1")
        elif prev_val==0 and curr_val==2:
            reward-=1
            #print("3. -5")
        elif prev_val==0 and curr_val==3:
            reward-=2
            #print("4. -10")
            
        elif prev_val==1 and curr_val==0:
            reward+=1
            #print("5. +5")
        elif prev_val==1 and curr_val==1:
            reward+=0.1
            #print("6. +1")
        elif prev_val==1 and curr_val==2:
            reward-=1
            #print("7. -5")
        elif prev_val==1 and curr_val==3:
            reward-=2
            
            #print("8. -10")
        elif prev_val==2 and curr_val==0:
            reward+=2
            #print("9. +10")
        elif prev_val==2 and curr_val==1:
            reward+=1
           # print("10. +5")
        elif prev_val==2 and curr_val==2:
            reward-=1
          #  print("11. -5")
        elif prev_val==2 and curr_val==3:
            reward-=2
         #   print("12. -10")
            
        #3->0 +10; 3->1 +5; 3->2 +1; 3->3 -10;
        elif prev_val==3 and curr_val==0:
            reward+=2
        #    print("13. +10")
        elif prev_val==3 and curr_val==1:
            reward+=1
       #     print("14. +5")
        elif prev_val==3 and curr_val==2:
            reward+=0.1
      #      print("15. +1")
        elif prev_val==3 and curr_val==3:
            reward-=2
     #       print("16. -10")
            
        # if any(curr ==3 for curr in current_state): #new changeeeeeee
        #     reward=0
        #     reward-=10
        #     break
        if action ==0 and current_state==[0,0,0,0,0]:
            reward+=2
            #print("17. +10 for moving forward with no obstacle")
    #print("REWARD IS ",reward)
    return reward


def train_dqn(model, target_model):
    batch_size = 64
    gamma = 0.99 ####changeeeeeee
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update = 20  # Update the target network every X episodes
    replay_buffer_size = 1000
    
    optimizer = optim.Adam(model.parameters()) #optimizer is the Adam optimizer 
    criterion = nn.MSELoss() #criterion is the Mean Squared Error (MSE) loss function

    qtarlist=[]
    qvalmean=[]
    replay_buffer = [] 
    actual_reward_list=[]
    rewards_list = []
    qsum_list = []
    q_values_list = []
    state=[]
    next_state=[]
    episodelist=[]
    losslist=[]
    epsilon_list=[]
    q_action_when_exploited=[]
    buffer_list=[]
    random_batch_list=[]
    increasing_epsilon=False #changed
    
    total_reward = 0
    for episode in range(num_episodes):
        robot.step(TIME_STEP)
        print("===========================================================================================")
        episodelist.append(episode)
        
        #time.sleep(1)
        print("Episode:", episode)

        state=read_sensors()

        
        
        #print("Current actual State:", state)
        bin_sensors1 = bin_sensor_values(state)

      #  print("Current binned State:", bin_sensors1)
       
        state_tensor = torch.tensor([bin_sensors1], dtype=torch.float32)

        action, q_action = select_action(state_tensor, epsilon)

       # print("Selected Action:", action)
        if action == 0:
            move_forward()
        elif action == 1:
            backward()
            #stop()
        elif action == 2:
            left()
        elif action == 3:
            right()

        print("q_action   ",q_action)
        #LOG THIS!!!!!!!!!
        
        # Execute the action and get the next state and reward 
        robot.step(TIME_STEP) # this is for updating the robot sensor values otherwise, the current and next states come up same
        next_state = read_sensors()
        bin_sensors2=bin_sensor_values(next_state)

        #print("Next actual State:", next_state)
        #print("Next binned State:", bin_sensors2)

        reward = calculate_reward(bin_sensors1,bin_sensors2, action) #calculate the reward based on current and next state

        #print("Reward:", reward)
        #print("===========================================================================================")
        
        total_reward += reward
        q_action_when_exploited.append(q_action)
        actual_reward_list.append(reward)
        next_state_tensor = torch.tensor([bin_sensors2], dtype=torch.float32)
        rewards_list.append(total_reward)
        # Store the transition in the replay buffer
        replay_buffer.append((state_tensor, action, reward, next_state_tensor))


        bin_sensors1 = bin_sensors2 #update
        if len(replay_buffer) > replay_buffer_size:
            replay_buffer.pop(0) #remove the first element from the replay buffer -- circular queue implementation

        # Train the DQN from the replay buffer
        if len(replay_buffer) >= 500: #NEEDS TO BE CHANGED TWEAKKKKKK #TWEAKED
            batch = np.random.choice(len(replay_buffer), batch_size, replace=False) #randomly select batch_size number of elements from the replay buffer
            random_batch_list.append(batch) #log into a new file
            #print("==============================================================")
            ##print("REPLAY BUFFER", replay_buffer)
            #print("==============================================================")
            state_batch, action_batch, reward_batch, next_state_batch = zip(*[replay_buffer[i] for i in batch]) #unzip the batch into individual lists

            state_batch = torch.cat(state_batch) #concatenate the state tensors in the batch 64x5
         #   print("State Batch Shape:", state_batch.shape)
            #print("State batch:",state_batch)
            action_batch = torch.tensor(action_batch) #batch of actions 64x1 ##whyyyy
          #  print("Action Batch Shape:", action_batch.shape)
            #print("Action batch", action_batch)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32) #batch of rewards 64x1
           # print("Reward Batch Shape:", reward_batch.shape)
            #print("reward batch", reward_batch)
            next_state_batch = torch.cat(next_state_batch) #concatenate the next state tensors in the batch 64x5
           # print("Next State Batch Shape:", next_state_batch.shape)
            #print("next_state_batch ",next_state_batch)

            q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)  #batch of max q values for current state 64x1
            q_values_target = target_model(next_state_batch).max(1)[0] #batch of max q values for next state 64x1 ##NEWLY CHNAGED
            q_sum = reward_batch + gamma * q_values_target  #calculate this for all 64 states for later use in the loss function #temporal difference
            
            # Clip Q-values and q_sum to prevent extreme values
            q_values = torch.clamp(q_values, -1e5, 1e5)
            q_sum = torch.clamp(q_sum, -1e5, 1e5)
            if episode == 500: #changeeeee #changed
                for i in range(500-1):
                    qtarlist_clone=q_sum.clone().detach()
                    q_values_list_clone=q_values.clone().detach()
                    qtarlist.extend(qtarlist_clone[i].item() for i in range(32))
                    q_values_list.extend(q_values_list_clone[i].item() for i in range(32))
                  
                    losslist.append(loss.item()) #whyyyyyy #why item()
                    qsum_list.append(q_sum.mean().item())
                    qvalmean.append(q_values.mean().item())
                print("qsum_list",qsum_list)
                print("qsum len",len(qsum_list))
                print("losslist",losslist)
                print("losslist",len(losslist))
                print("qtar list", qtarlist)
                print("qtar len: ", len(qtarlist))
                print("q_values_list",q_values_list)
                print("qvallist",len(q_values_list))
            print("QSUM :",q_sum)
            # for i in range(32):
                # qtarlist.append(q_sum[i])
                # q_values_list.append(q_values[i])
            qtarlist_clone=q_sum.clone().detach()
            q_values_list_clone=q_values.clone().detach()

            qtarlist.extend(qtarlist_clone[i].item() for i in range(32))
            q_values_list.extend(q_values_list_clone[i].item() for i in range(32))

            print("QSUM MEAN: ",q_sum.mean())
            qsum_list.append(q_sum.mean().item()) 
            qvalmean.append(q_values.mean().item())
            # Check for NaN or infinity in inputs
            if torch.isnan(q_values).any() or torch.isnan(q_sum).any():
               print("NaN or Infinity detected in Q-values or q_sum. Skipping update.")
               continue
            #print("QSUM CALCULATED")
            #print("QVAL",q_values)
            #print("QSUM",q_sum)



            # Logging information for each batch into the log file
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Episode: {episode}======================\n")
                for i in range(batch_size):
                    log_entry = f"Current State: {state_batch[i]}, " \
                                f"Action: {action_batch[i]}, " \
                                f"Next State: {next_state_batch[i]}, " \
                                f"Reward: {reward_batch[i]}, " \
                                f"Q Source Value: {q_values[i]}, " \
                                f"Q Target Value: {q_values_target[i]}, " \
                                f"Q Target_Bellman: {q_sum[i]}\n"
                    log_file.write(log_entry)



            loss = criterion(q_values, q_sum) #calculate the loss between q_values and q_sum temporal difference
            #print("LOSS CALCULATED", loss.item())
            
           
        
            losslist.append(loss.item()) ######whyyyyy item()
            
            optimizer.zero_grad() #zero the gradients because PyTorch accumulates the gradients on subsequent backward passes
            loss.backward() #backpropagate the loss
            #nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    

            optimizer.step() #update the weights and biases
            #print("WEIGHTS ARE UPDATED")
            #this is the q_values tensor #append the q_values tensor to the q_values_list
            #q_values_list.append(q_values.detach().numpy().astype(object))
            #append the mean of the q_values tensor to the q_values_list
            #qsum_list.append(q_sum.mean().item()) #append the mean of the q_sum tensor to the qsum_list
            #print("qsum mean",q_sum.mean().item())
            #print("qsum list",qsum_list)
            buffer_list.append(replay_buffer)
            
        #EPSILON SHOULD OSCILLATE BETWEEN 0.1 TO 0.99
         #0.995 ?? decay value?? whyyyyyyyyyy
        #print("epsilon value",epsilon)
        if not increasing_epsilon: #changed
            epsilon = max(epsilon * epsilon_decay, 0.1)
            if epsilon<=0.1:
                increasing_epsilon=True
        else:
            epsilon=min(epsilon/epsilon_decay,0.99)
            if epsilon>=0.99:
                increasing_epsilon=False
        # if epsilon==0.1:
        #     epsilon=0.99
        # #print("EPSILON IS DECAYED")
        epsilon_list.append(epsilon)

        # Update the target network
        if episode % target_update == 0 and episode>=500: #changed since before 500, not needed to transfer
            target_model.load_state_dict(model.state_dict())

  
    return episodelist, actual_reward_list, rewards_list, qsum_list, q_values_list,losslist, epsilon_list, buffer_list, random_batch_list, q_action_when_exploited, qtarlist, qvalmean

init_tarpy()

#print("Why is robot not executing for so long??")
#if __name__ == "__main__": #this is executed only if the file is run directly and not imported #good coding practice in Python     
input_size = 5
output_size = 4  # Assuming four possible actions (forward, backward, left, right)
num_episodes = 15000

      
model = DQN(input_size, output_size) #model is the DQN instance
target_model = DQN(input_size, output_size) #target_model is the target DQN instance
target_model.load_state_dict(model.state_dict()) #copy the weights and biases from model to target_model #NEWLY CHANGED #####whattt
target_model.eval() #set target_model to evaluation mode #this is to freeze the target_model while training the model ###whyyy

print("Starting Training...")
episodelist, actual_rewards, cum_rewards, qsum, q_values,losslist,epsilon_list, buffer_list, random_batch_list, q_action_when_exploited, qtarlist, qvalmean = train_dqn(model, target_model)
print("Training Complete!")
# Save rewards, qsum, and q_values to text files
rewdata = np.column_stack((episodelist, actual_rewards, epsilon_list)) #combine both the lists into a single 2D array
# Save the combined data to the file "actual_rewards.txt"
np.savetxt("actual_rewards.txt", rewdata, fmt="%d %.4f %.4f", header=" Episode Actual Epsilon", comments="")

cumrewdata = np.column_stack((episodelist, cum_rewards, epsilon_list)) #combine both the lists into a single 2D array
np.savetxt("cum_rewards.txt", cumrewdata, fmt="%d %.4f %.4f", header="Episode Cumulative Epsilon", comments="")

qsumdata= np.column_stack((episodelist, qsum, qvalmean, epsilon_list)) #combine both the lists into a single 2D array
np.savetxt("qtargetmean_VS_qsrcmean.txt", qsumdata, fmt="%d %.4f %.4f %.4f", header="Episode Qtargetmean Qsrcmean Epsilon", comments="")
# Convert each tensor to a NumPy array and store them in a list
# q_values = np.column_stack((episodelist, q_values, epsilon_list))
# np.savetxt("Weights.txt", q_values, fmt="%s", header="Episode Number,  Weights", comments="")

lossvalues = np.column_stack((episodelist, losslist, epsilon_list)) #combine both the lists into a single 2D array
np.savetxt("loss.txt", lossvalues, fmt="%d %.4f %.4f", header="Episode Loss Epsilon", comments="")
#PLOTTING
#qtarlistval = np.column_stack((q_values, np.array(qtarlist)))


qtarlistval=np.column_stack((q_values, qtarlist))
np.savetxt("qsrc_vs_qtarget_indiv.txt", qtarlistval,fmt="%s %s", header="qvalues qtarget",comments="")
# bufferval=np.column_stack((episodelist, buffer_list))
# np.savetxt("buffer.txt",bufferval,fmt="%d %.4f", header="Episode BUFFER") ###guess thos is where i get the error becz the bufferlist is again a list ##

# qaction=np.column_stack((episodelist,q_action_when_exploited))



episode_numbers = rewdata[:, 0]
actual_rewards = rewdata[:, 1]
# Plot the data as a scatter plot
plt.figure(figsize=(10,6))
plt.scatter(episode_numbers, actual_rewards, marker='o',s=10)
plt.xlabel('Episode Number')
plt.ylabel('Actual Rewards')
plt.title('Episode Number vs. Actual Rewards')
plt.grid(True)
plt.savefig("actual_rewards_chart.png",dpi=300)
#plt.show()

# Extract episode numbers and cumulative rewards from the data
episode_numbers = cumrewdata[:, 0]
cumulative_rewards = cumrewdata[:, 1]
plt.figure(figsize=(10,6))
plt.plot(episode_numbers, cumulative_rewards, marker='o', linestyle='-',linewidth=1.5)

plt.xlabel('Episode Number')
plt.ylabel('Cumulative Rewards')
plt.title('Episode Number vs. Cumulative Rewards')
plt.grid(True)
plt.savefig("cum_rewards_chart.png",dpi=300)
#plt.show()

# Extract episode numbers and qsum from the data
episode_numbers = qsumdata[:, 0]
qsum = qsumdata[:, 1]
plt.figure(figsize=(10,6))
plt.plot(episode_numbers, qsum, marker='o', linestyle='-',linewidth=1.5)

plt.xlabel('Episode Number')
plt.ylabel('Q-value Mean')
plt.title('Episode Number vs. Q-value Mean')
plt.grid(True)
plt.savefig("qmean_chart.png",dpi=300)
#plt.show()
# Extract episode numbers and loss from the data
episode_numbers = lossvalues[:, 0]
loss = lossvalues[:, 1]
plt.figure(figsize=(10,6))
plt.plot(episode_numbers, loss, marker='o', linestyle='-', linewidth=1.5)

plt.xlabel('Episode Number')
plt.ylabel('Loss')
plt.title('Episode Number vs. Loss')
plt.grid(True)
plt.savefig("loss_chart.png",dpi=300)
#plt.show()

tar.keep_alive()
