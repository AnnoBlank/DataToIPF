# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:05:27 2024

@author: Robin
"""
from datetime import datetime
import pytz

now = datetime.now(pytz.utc)
nowstr = (str(now.date())+'_'+
          "{:02}".format(now.hour)+'h_'+
                    "{:02}".format(now.minute)+'m_'+
                              "{:02}".format(now.second)+'s')
print(now.date())
print(now.hour)
print(now.minute)
print(type(now.second))

print(nowstr)