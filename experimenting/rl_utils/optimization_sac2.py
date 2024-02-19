import torch
import torch.nn as nn

def perform_optimization_step(actor, critic_1, critic_2, critic_1_target, critic_2_target, alpha_factor,
                              minibatch, gamma, tau, device, grad_clip_value=10, curriculum=False):
    if curriculum:
        state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d, td) in minibatch])
        action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.float32) for (s, a, r, s_, d, td)  in minibatch])
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d, td)  in minibatch])
        next_state_batch = torch.cat([torch.tensor([s_], device=device, dtype=torch.float32) for (s, a, r, s_, d, td)  in minibatch])
        done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d, td)  in minibatch])
    else:
        state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        next_state_batch = torch.cat([torch.tensor([s_], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
        done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d) in minibatch])

    actions, log_probs, _, _ = actor.sample_action(state_batch, reparameterize=True)
    log_probs = log_probs.view(-1)

# Updating the alpha scale factor
    alpha_factor.alpha = alpha_factor.log_alpha.exp().detach()
    alpha_loss = - (alpha_factor.log_alpha * (log_probs + alpha_factor.target_entropy).detach()).mean()
    alpha_factor.optimizer.zero_grad()
    alpha_loss.backward()
    alpha_factor.optimizer.step()

# Updating the critics
    with torch.no_grad():
        next_actions, next_log_probs, _, _ = actor.sample_action(next_state_batch, reparameterize=True)
        next_log_probs = next_log_probs.view(-1)
        q1_next_target = critic_1_target(next_state_batch, next_actions)
        q2_next_target = critic_2_target(next_state_batch, next_actions)
        min_next_value = torch.min(q1_next_target, q2_next_target).view(-1) # Taking the minimum is proposed in the TD3 paper to deal with overestimation bias
        min_next_value -= alpha_factor.alpha * next_log_probs
        q_hat = reward_batch + (1 - done_batch * 1) * (min_next_value * gamma)
    q1 = critic_1(state_batch, action_batch).view(-1)
    q2 = critic_2(state_batch, action_batch).view(-1)
    critic_loss_1 = 0.5 * torch.nn.functional.mse_loss(q1, q_hat)
    critic_loss_2 = 0.5 * torch.nn.functional.mse_loss(q2, q_hat)
    critic_loss = critic_loss_1 + critic_loss_2
    critic_1.optimizer.zero_grad()
    critic_2.optimizer.zero_grad()
    critic_loss.backward()
    #torch.nn.utils.clip_grad_norm_(critic_1.parameters(), grad_clip_value)
    #torch.nn.utils.clip_grad_norm_(critic_2.parameters(), grad_clip_value)
    critic_1.optimizer.step()
    critic_2.optimizer.step()

    # Updating the actor
    q1_actor = critic_1(state_batch, actions)
    q2_actor = critic_2(state_batch, actions)
    min_value = torch.min(q1_actor, q2_actor) # Proposed in the TD3 paper to deal with overestimation bias
    actor_loss = (alpha_factor.alpha * log_probs - min_value).mean()
    actor.optimizer.zero_grad()
    actor_loss.backward()
    #torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_value)
    actor.optimizer.step()

    # Update target networks
    # Network 1
    critic_1_weights = critic_1.state_dict()
    critic_1_target_weights = critic_1_target.state_dict()
    for set in critic_1_weights:
        critic_1_target_weights[set] = tau * critic_1_weights[set].clone() + (1 - tau) * critic_1_target_weights[set].clone()
    critic_1_target.load_state_dict(critic_1_target_weights)
    # Network 2
    critic_2_weights = critic_2.state_dict()
    critic_2_target_weights = critic_2_target.state_dict()
    for set in critic_2_weights:
        critic_2_target_weights[set] = tau * critic_2_weights[set].clone() + (1 - tau) * critic_2_target_weights[set].clone()
    critic_2_target.load_state_dict(critic_2_target_weights)
    
    return actor_loss.item(), critic_loss_1.item(), critic_loss_2.item(), alpha_loss.item(), q1.mean().item(), q2.mean().item(), alpha_factor.alpha.item() 