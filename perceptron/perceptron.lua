require 'torch'
require 'nn'
require 'Sign'
require 'SignCriterion'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Perceptron Learning Algorithm')
cmd:text()
cmd:text('Options:')
cmd:option('-N', 10, 'number of training samples')
cmd:option('-T', 1000, 'number of trials')
cmd:option('-plot', false, 'plot learning problem, live')
cmd:option('-verbose', false, 'print intermediate info')
cmd:text()
if arg then opt = cmd:parse(arg) else opt = cmd:parse({}) end

--- Returns a random point in the space [-1,1]x[-1,1].
function rand_point()
   return (torch.rand(2)*2)-1
end

--- Returns a random line passing through two random points.
function rand_line()
   local p1 = rand_point()
   local p2 = rand_point()
   local a = (p2[2] - p1[2])/(p2[1] - p1[1])
   local b = p1[2] - a*p1[1]
   local l = torch.DoubleTensor({b, a, 1})
   return l
end

--- Returns a perceptron, with weights initialized according to the
--- given line or the line x2=0 by default.
function perceptron(l)
   l = l or torch.DoubleTensor({0, 0, 1})
   local p = nn.Sequential()
   p:add(nn.Linear(2, 1))
   p:add(Sign())
   local ws, _ = p:parameters()
   ws[1][1][1] = l[2]
   ws[1][1][2] = -l[3]
   ws[2][1] = l[1]
   return p
end

--- Returns the parameters of the perceptron.
function perceptron_params(p)
   local ws, _ = p:parameters()
   local a = ws[1][1][1]
   local d = -ws[1][1][2]
   local b = ws[2][1]
   return b, a, d
end

--- Returns the parameters of the perceptron as a vector.
function perceptron_vec(p)
   local b, a, d = perceptron_params(p)
   return torch.DoubleTensor({b, a, d})
end

--- Returns a function from x1 to x2, which represents the perceptron
--- separation line. Also returns a vector of x1 in [-1;1] such that
--- f(x1) in [-1;1] (assuming no overflows).
function perceptron_separation_line(p)
   local b, a, d = perceptron_params(p)
   local x1_1, x1_2 = (-d-b)/a, (d-b)/a
   local min_x1 = math.max(-1, math.min(x1_1, x1_2))
   local max_x1 = math.min(1,  math.max(x1_1, x1_2))
   if (max_x1 < min_x1) then
      min_x1 = -1
      max_x1 = 1
   end
   return (function(x1) return (x1*a+b)/d end), torch.linspace(min_x1, max_x1)
end

function sample_positive(si)
   return si[2]:squeeze() > 0
end

function sample_x1(si)
   return si[1][1]
end

function sample_x2(si)
   return si[1][2]
end

--- Returns N training samples determined by the given perceptron.
function rand_samples(N, p)
   local s = {}
   local np = 0
   while (np == 0 or np == N) do
      s = {}
      np = 0
      for i=1,N do
         local x = rand_point()
         local y = p:forward(x):clone()
         s[i] = {x,y}
         if sample_positive(s[i]) then np = np + 1 end
      end
   end
   return s
end

--- Partition the samples according to the given predicate.
function partition_samples(s, predicate)
   local ps = {}
   local ns = {}
   for _,si in ipairs(s) do
      if (predicate(si)) then
         table.insert(ps, si)
      else
         table.insert(ns, si)
      end
   end
   return ps, ns
end

-- Returns two vectors, one for each coordinate.
function coordinates_of_samples(s)
   local n = table.getn(s)
   local indexed = function(f)
      return (function(i) return f(s[i]) end)
   end
   local x1s = torch.range(1, n+.1):apply(indexed(sample_x1))
   local x2s = torch.range(1, n+.1):apply(indexed(sample_x2))
   return x1s, x2s
end

--- Visualize the learning problem by plotting the sample points and
--- the perceptron separation lines.
function viz(s, p1, p2)
   local ps, ns = partition_samples(s, function(si) return sample_positive(si) end)
   local x1p, x2p = coordinates_of_samples(ps)
   local x1n, x2n = coordinates_of_samples(ns)
   local f1, x11 = perceptron_separation_line(p1)
   local f2, x12 = perceptron_separation_line(p2)
   gnuplot.plot({x1n, x2n}, {x1p, x2p}, {x11, f1(x11), '-'}, {x12, f2(x12), '-'})
end

--- Train the perceptron on a randomly generated linearly separable
--- problem with N samples. Returns the number of iterations and the
--- probability of missclassification based on the angle between the
--- separation lines.
function train(N, plot)
   local p_true = perceptron(rand_line())
   local s = rand_samples(N, p_true)
   local p_learn = perceptron()

   if plot then viz(s, p_true, p_learn) end

   local criterion = SignCriterion()

   local is_missclassified = function(si)
      local pred = p_learn:forward(si[1])
      return criterion:forward(pred, si[2]) > 0
   end
   local all_missclassified = function()
      local missed_s, _ = partition_samples(s, is_missclassified)
      return missed_s, table.getn(missed_s)
   end

   local num_iter = 0
   local missed_s, num_missed = all_missclassified()
   while (num_missed > 0) do
      local ri = torch.random(num_missed)
      local si = missed_s[ri]

      criterion:forward(p_learn:forward(si[1]), si[2])
      p_learn:zeroGradParameters()
      p_learn:backward(si[1], criterion:backward(p_learn.output, si[2]))
      p_learn:updateParameters(1)

      num_iter = num_iter + 1
      if plot then viz(s, p_true, p_learn) end
      missed_s, num_missed = all_missclassified()
   end

   local v_true = perceptron_vec(p_true)
   local v_learn = perceptron_vec(p_learn)
   local miss_prob =
      torch.acos(torch.abs(v_true:dot(v_learn))/(v_true:norm()*v_learn:norm()))/
      math.pi

   if (opt.verbose) then
      print('#' .. tostring(num_iter) .. ':' .. tostring(miss_prob))
   end

   return num_iter, miss_prob
end

--- Run a perceptron learning experiment with T trials and N samples each time.
--- Returns and prints the average number of iterations and the
--- average probability of missclassfication per trial.
function experiment(T, N)
   local avg_iter = 0
   local avg_miss_prob = 0
   for ti=1,T do
      if (opt.verbose) then
         print('-- trial ' .. tostring(ti) .. ' --')
      end
      local num_iter, miss_prob = train(N, opt.plot)
      avg_iter = avg_iter + num_iter/T
      avg_miss_prob = avg_miss_prob + miss_prob/T
   end
   print('results for ' .. tostring(T) .. ' trials and ' .. tostring(N) .. ' samples')
   print('average number of iterations: ' .. tostring(avg_iter))
   print('average probability of missclassifcation: ' .. tostring(avg_miss_prob))
   return avg_iter, avg_miss_prob
end

function main()
   experiment(opt.T, opt.N)
end
if arg then main() end