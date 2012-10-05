local SignCriterion, parent = torch.class('SignCriterion', 'nn.Criterion')

function SignCriterion:__init()
   parent.__init(self)
   self.gradInput = torch.Tensor(1)
end

function SignCriterion:updateOutput(input, target)
   if ((input*target) > 0) then
      self.output = 0
   else
      self.output = 1
   end
   return self.output
end

function SignCriterion:updateGradInput(input, target)
   if ((input*target) > 0) then
      self.gradInput[1] = 0
   else
      self.gradInput[1] = input[1]
   end
  return self.gradInput
end
