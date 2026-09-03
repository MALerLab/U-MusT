import torch
import torch.nn.functional as F

def top_p_sampling(logits, thres=0.9):
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

  sorted_indices_to_remove = cum_probs > thres
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0


  remove_mask = torch.zeros_like(logits, dtype=torch.bool)
  remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

  new_logits = logits.clone()
  new_logits[remove_mask] = float('-inf')
  return new_logits

  # Create an empty tensor to hold the new logits
  new_logits = logits.clone()

  # Use the sorted indices to place the '-inf' in the original places
  indices_to_remove = sorted_indices[sorted_indices_to_remove]
  new_logits[..., indices_to_remove] = float('-inf')
  return new_logits

# refered: https://github.com/cimeister/typical-sampling
def typical_sampling(logits, thres=0.99):
  # calculate entropy
  normalized = torch.nn.functional.log_softmax(logits, dim=-1)
  p = torch.exp(normalized)
  ent = -(normalized * p).nansum(-1, keepdim=True)

  # shift and sort
  shifted_scores = torch.abs((-normalized) - ent)
  sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
  sorted_logits = logits.gather(-1, sorted_indices)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

  # Remove tokens with cumulative mass above the threshold
  last_ind = (cumulative_probs < thres).sum(dim=-1)
  last_ind[last_ind < 0] = 0
  sorted_indices_to_remove = sorted_scores > sorted_scores.gather(-1, last_ind.view(-1, 1, 1))
  # if self.min_tokens_to_keep > 1:
  #     # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
  #     sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
  indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)

  scores = logits.masked_fill(indices_to_remove, float("-inf"))
  return scores

# refered: https://github.com/john-hewitt/truncation-sampling
def eta_sampling(logits, epsilon) -> torch.FloatTensor:
  probabilities = logits.softmax(dim=-1)
  entropy = torch.distributions.Categorical(probs=probabilities).entropy()
  new_epsilon = min(epsilon, torch.sqrt(torch.tensor(epsilon))*torch.exp(-entropy))
  indices_to_remove = probabilities < new_epsilon
  max_word = torch.argmax(logits, dim=-1)
  indices_to_remove[..., max_word.squeeze()] = 0
  new_scores = logits.masked_fill(indices_to_remove, float("-inf"))
  return new_scores

def sample(logits, sampling_method, threshold, temperature):
  """
    Sample from the logits with a specific sampling strategy.
    Args:
      logits: (batch_size, seq_len, vocab_size)
      sampling_method: str, sampling method
      threshold: float, threshold for sampling
      temperature: float, temperature for sampling
    Returns:
      sampled_tokens: (batch_size, 1) # CAUTION: The function is modified to return batch samples
  """
  if sampling_method == "top_p":
    probs = F.softmax(top_p_sampling(logits, thres=threshold) / temperature, dim=-1)
  elif sampling_method == "typical":
    probs = F.softmax(typical_sampling(logits, thres=threshold) / temperature, dim=-1)
  elif sampling_method == "eta":
    probs = F.softmax(eta_sampling(logits, epsilon=threshold) / temperature, dim=-1)
  elif sampling_method == "argmax":
    return torch.argmax(logits, -1)
  else:
    probs = F.softmax(logits / temperature, dim=-1)
  if torch.isnan(probs).any() or torch.isinf(probs).any():
    print(f"nan or inf in probs: {probs}")
  return torch.multinomial(probs[:,-1,:], 1)
