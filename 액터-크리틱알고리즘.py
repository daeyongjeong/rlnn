#  액터-크리틱알고리즘
np.random.seed(0)

# 환경, 에이전트를 초기화
env = Environment()
agent = Agent()
gamma = 0.9

# p(𝑠,𝑎), 𝑉(𝑠)←임의의 값
V = np.random.rand(env.reward.shape[0], env.reward.shape[1])
policy = np.random.rand(env.reward.shape[0], env.reward.shape[1],len(agent.action))
# 확률의 합이 1이 되도록 변환
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        policy[i,j,:] = policy[i,j,:] /np.sum(policy[i,j,:])

max_episode = 10000
max_step = 100

print("start Actor-Critic")
alpha = 0.1

# 각 에피소드에 대해 반복 :
for epi in tqdm(range(max_episode)):
    # S 를 초기화
    i = 0
    j = 0
    agent.set_pos([i,j])

    # 에피소드의 각 스텝에 대해 반복 :
    for k in range(max_step):
        
        # Actor : p(𝑠,𝑎)로 부터 a를 선택 ( 예 :Gibbs softmax method)
        pos = agent.get_pos()
        
        # Gibbs softmax method 로 선택될 확률을 조정
        pr = np.zeros(4)
        for i in range(len(agent.action)):
            pr[i] = np.exp(policy[pos[0],pos[1],i])/np.sum(np.exp(policy[pos[0],pos[1],:]))
                
        # 행동 선택
        action =  np.random.choice(range(0,len(agent.action)), p=pr)            
        
        # 행동 a를 취한 후 보상 r과 다음 상태 s'를 관측
        observation, reward, done = env.move(agent, action)
        
        # Critic 학습
        # δt=r(t+1)+γV(S(t+1) )-V(St)
        td_error = reward + gamma * V[observation[0],observation[1]] - V[pos[0],pos[1]]
        V[pos[0],pos[1]] += alpha * td_error

        # Actor 학습 :
        # p(st,at)=p(st,at)-βδ_t
        policy[pos[0],pos[1],action] += td_error * 0.01
        
        # 확률에 음수가 있을경우 전부 양수가 되도록 보정
        if np.min(policy[pos[0],pos[1],:]) < 0:
            policy[pos[0],pos[1],:] -= np.min(policy[pos[0],pos[1],:])
        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                policy[i,j,:] = policy[i,j,:] /np.sum(policy[i,j,:])
        
        # s가 마지막 상태라면 종료
        if done == True:
            break

# 학습된 정책에서 최적 행동 추출
optimal_policy = np.zeros((env.reward.shape[0],env.reward.shape[1]))
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        optimal_policy[i,j] = np.argmax(policy[i,j,:])
        
print("Actor - Critic : V(s)")
show_v_table(np.round(V,2),env)
print("Actor - Critic : policy(s,a)")
show_q_table(np.round(policy,2),env)
print("Actor - Critic : optimal policy")
show_policy(optimal_policy,env)   