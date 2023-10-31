import torch

def perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device, batch_size, curriculum=False):
    
    if curriculum:
        state_batch = torch.cat([s for (s, a, r, s_, d, td) in minibatch])
        action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.int64) for (s, a, r, s_, d, td)  in minibatch]).reshape([batch_size,1])
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d, td)  in minibatch])
        next_state_batch = torch.cat([s_ for (s, a, r, s_, d, td)  in minibatch])
        done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d, td)  in minibatch])
    else:
        state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.int64) for (s, a, r, s_, d) in minibatch]).reshape([batch_size,1])
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        next_state_batch = torch.cat([torch.tensor([s_], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d) in minibatch])

    state_action_q_values_policy = model_policy(state_batch).gather(1, action_batch)
    
    with torch.no_grad():
        state_action_q_values_target = model_target(next_state_batch).max(1)[0].detach()

    expected_state_action_q_values = reward_batch + (1 - done_batch * 1) * (state_action_q_values_target * gamma)

    loss = criterion(state_action_q_values_policy, expected_state_action_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_policy.parameters(), 100) 
    optimizer.step()
    
    return loss.item()