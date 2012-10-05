local Sign = torch.class('Sign', 'nn.Module')

function Sign:updateOutput(input)
   self.output:resizeAs(input):copy(input):apply(
      function(x)
         if (x > 0) then return 1 else return -1 end
      end)
   return self.output
end

function Sign:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   return self.gradInput
end
