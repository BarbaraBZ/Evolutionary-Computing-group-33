exp7 <- gain_all_tidy[gain_all_tidy$enemy==7,]
boxplot(gain~exp, data=exp7)
exp7gen <- exp7[exp7$exp=='gen',]
exp7steady <- exp7[exp7$exp=='steady',]
t.test(exp7gen$gain, exp7steady$gain)
