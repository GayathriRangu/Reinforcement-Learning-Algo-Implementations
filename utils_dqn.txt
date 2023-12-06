##CODE FOR TESTING

def testing_model(test_model, test_episodes,modelnum,runnum):
    t1=time.time()
    print("testing model")
    collision_count=0
    success_count=0
    window_size_t=50
    reward_window_t = []
    reward_moving_avg_t=[]
    epsilon=0.05
    with open(f"testing_ACTUAL_REW_{modelnum}_test_{runnum}",'w') as tfile:
        tfile.write("Episode\t")
        tfile.write("Actual_Reward\t")
        tfile.write("Time\n")
    with open(f"testing_CUM_REW_{modelnum}_test_{runnum}",'w') as tfile:
        tfile.write("Episode\t")
        tfile.write("Cumulative_Reward\t")
        tfile.write("Success_count\t")
        tfile.write("Collision count\t")
        tfile.write("Time\n")
    with open(f"testing_Qvalues_{modelnum}_test_{runnum}",'w') as tfile:
        tfile.write("Episode\t")
        tfile.write("Q1\t\t")
        tfile.write("Q2\t\t")
        tfile.write("Q3\t\t")
        tfile.write("Q4\t\t")
        tfile.write("Qmax\t\t")
        tfile.write("Action\t\t")
        tfile.write("Reward\t\t")
        tfile.write("Cumulative_Reward\t")
        tfile.write("Time\n")
        total_reward = 0
        actual_rewards=[]
    for episode in range(test_episodes):
        robot.step(TIME_STEP)
        state = read_sensors()
        print("state ",state)
        bin_sensors1 = bin_sensor_values(state)
        print("bin_sensor1 ",bin_sensors1)
        print("satte check ",state)
        state_tensor = torch.tensor([normalized_sensors1], dtype=torch.float32)
        action, q_action = select_action(state_tensor, epsilon,test_model)  ##NOTE: CHANGE THE SELECT_ACTION FUNCTION, PASS THE TEST_MODEL PARAMETER.
        if action == 0:
            move_forward()
        elif action == 1:
            backward()
            #stop()
        elif action == 2:
            left()
        elif action == 3:
            right()

        print("action ",action)
        print("qaction ",q_action)
        t2=time.time()
        t3=t2-t1

        reward_moving_avg_t=0
        with torch.no_grad():#since we are not training the model, we dont need to calculate the gradients hence using torch.no_grad()
            q_values_t = test_model(state_tensor)
        robot.step(TIME_STEP) # this is for updating the robot sensor values otherwise, the current and next states come up same
        next_state = read_sensors()
        bin_sensors2=bin_sensor_values(next_state)
        reward = calculate_reward(bin_sensors1,bin_sensors2, action) 
        print("reward ",reward)
        actual_rewards.append(reward)
        total_reward += reward 
        state = copy.copy(next_state)
        if len(reward_window_t)==window_size_t:
            reward_window_t.pop(0)
        reward_window_t.append(reward)
        reward_moving_avg_t=sum(reward_window_t)
        with open(f"testing_ACTUAL_REW_{modelnum}_test_{runnum}",'a') as tfile:
            tfile.write(str(episode))
            tfile.write("\t")
            tfile.write(str(reward))
            tfile.write("\t")
            tfile.write(str(t3))
            tfile.write("\n")
        with open(f"testing_CUM_REW_{modelnum}_test_{runnum}",'a') as tfile:
            tfile.write(str(episode))
            tfile.write("\t")
            tfile.write(str(reward_moving_avg_t))
            tfile.write("\t")
            tfile.write(str(success_count))
            tfile.write("\t")
            tfile.write(str(collision_count))
            tfile.write("\t")
            tfile.write(str(t3))
            tfile.write("\n")

            tfile.write(str(torch.max(q_values_t).item()))
            tfile.write("\t")
            tfile.write(str(action))
            tfile.write("\t")
            tfile.write(str(reward))
            tfile.write("\t")
            tfile.write(str(reward_moving_avg_t))
            tfile.write("\t")
            tfile.write(str(t3))
            tfile.write("\n")
            print(f"============{episode}=================")
            print(f"State {state}\t Action: {action} \t Next State: {next_state} \t Reward: {reward}")
            print(f"Q-values: {q_values_t}")
            print(f"Qmax: {torch.max(q_values_t)}")


test_model = DQN(input_size, output_size) 
#MODIFY THIS LIST BASED ON THE MODELS YOU WANT TO TEST--HERE TESTING 5 MODELS FOR THREE TIMES EACH
modellist=[24000,25000,26000,27000,28000,29000]
for modelnum in modellist:
    for runnum in range(3):
    # Load the weights into the model
        checkpoint = torch.load(f'model_weights_episode_{modelnum}.pt')
        test_model.load_state_dict(checkpoint)
        # Set the model to evaluation mode
        test_model.eval()
        test_ep=3000
        trial=1
        testing_model(test_model,test_ep,modelnum,runnum) #Calling the test fn
    print("Done testing!!")


#######################CODE FOR SAVING THE MODEL DURIN TRAINING################
# Function to save model weights to a file
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)

if episode%500==0: #INCLUDE THIS AT THE END OF TRAINING FOR LOOP
    save_qcon_table(q_con_table,episode)