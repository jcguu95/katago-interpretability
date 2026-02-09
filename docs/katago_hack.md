         outp = self.bias2(outp, mask=mask, mask_sum=mask_sum)
         outp = self.act2(outp)
         if extra_outputs is not None:
             extra_outputs.report("policy_penultimate", outp)
         outp = self.conv2p(outp)
 =======
         outp = self.bias2(outp, mask=mask, mask_sum=mask_sum)
         outp = self.act2(outp)
         if extra_outputs is not None:
             extra_outputs.report("policy_penultimate", outp)
         outp = self.conv2p(outp)
 >>>>>>> REPLACE
