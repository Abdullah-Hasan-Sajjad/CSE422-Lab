from pickle import FALSE, TRUE

#################
# taking input
#################

#taking number of transections 

transection_amount_input=-1
transection_amount_input_accepted=FALSE

while   (transection_amount_input<1   or transection_amount_input>10**2)   and transection_amount_input_accepted==FALSE:

    transection_amount_input =int(input())

    if  transection_amount_input>=1   and transection_amount_input<=10**2:
        transection_amount_input_accepted=TRUE

    else:
        print('invalid number. N( 1 ≤ N ≤ 10^2 )')


value_accepted=FALSE

#taking transection values

transections_input=[]

for i   in  range(transection_amount_input):

    value_accepted=FALSE

    while   value_accepted==FALSE:

        transection=input().replace(' ', '')

        if  transection[0]=='l':
            
            value=0-int(transection[1:])
            transections_input.append(value)
            value_accepted=TRUE

        elif    transection[0]=='d':

            value=0+int(transection[1:])
            transections_input.append(value)
            value_accepted=TRUE

        else:
            print('wrong input, give again')
            value_accepted=FALSE

    
    

if(value_accepted==TRUE):
    pass
    #code
else:
    print('wrong input given')

