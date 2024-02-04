import torch
import torch.nn as nn

def perform_optimization_step(actor, critic_1, critic_2, v_1, v_2_target, minibatch, gamma, tau, temperature_factor, device, batch_size, grad_clip_value=100, curriculum=False):
    if curriculum:
        state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d, td) in minibatch])
        action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.int64) for (s, a, r, s_, d, td)  in minibatch]).reshape([batch_size,1])
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d, td)  in minibatch])
        next_state_batch = torch.cat([torch.tensor([s_], device=device, dtype=torch.float32) for (s, a, r, s_, d, td)  in minibatch])
        done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d, td)  in minibatch])
    else:
        state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.int64) for (s, a, r, s_, d) in minibatch]).reshape([batch_size,1])
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        next_state_batch = torch.cat([torch.tensor([s_], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d) in minibatch])

    def inner_function_for_critic(state, reparameterize):
        actions, log_probs = actor.sample_action(state, reparameterize=reparameterize)
        log_probs = log_probs.view(-1)
        q1 = critic_1(state, actions)
        q2 = critic_2(state, actions)
        critic_value = torch.min(q1, q2) # Proposed in the TD3 paper to deal with overestimation bias
        critic_value = critic_value.view(-1)
        return critic_value, log_probs

    value = v_1(state_batch).view(-1)
    value_next_state = v_2_target(next_state_batch).view(-1)

    # Updating the value function
    critic_value, log_probs = inner_function_for_critic(state_batch, reparameterize=False)
    value_target = critic_value - log_probs
    value_loss = 0.5 * torch.nn.functional.mse_loss(value, value_target.detach())
    v_1.optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(v_1.parameters(), grad_clip_value)
    v_1.optimizer.step()

    # Updating the actor
    critic_value, log_probs = inner_function_for_critic(state_batch, reparameterize=False)
    actor_loss = (log_probs - critic_value).mean()
    actor.optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_value)
    actor.optimizer.step()

    # Updating the critics
    q_hat = temperature_factor * reward_batch + (1 - done_batch * 1) * (value_next_state * gamma)
    q_1 = critic_1(state_batch, action_batch).view(-1)
    q_2 = critic_2(state_batch, action_batch).view(-1)
    critic_loss_1 = torch.nn.functional.mse_loss(q_1, q_hat.detach())
    critic_loss_2 = torch.nn.functional.mse_loss(q_2, q_hat.detach())
    critic_loss = critic_loss_1 + critic_loss_2
    critic_1.optimizer.zero_grad()
    critic_2.optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(critic_1.parameters(), grad_clip_value)
    #torch.nn.utils.clip_grad_norm_(critic_2.parameters(), grad_clip_value)
    critic_1.optimizer.step()
    critic_2.optimizer.step()

    # Update target network
    value_weights = v_1.state_dict()
    target_weights = v_2_target.state_dict()
    for name in value_weights:
        target_weights[name] = tau * value_weights[name].clone() + (1 - tau) * target_weights[name].clone()
    v_2_target.load_state_dict(target_weights)