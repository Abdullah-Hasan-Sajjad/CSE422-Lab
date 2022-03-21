def dfs(row,column,count):

    #checking (x, y+1) right position for child

    try:
        
        if table[row][column+1]=='Y':

            if  str(row)+str(column+1) in visited :
                pass

            else:
                visited.append(str(row)+str(column+1))
                Stack.append(str(row)+str(column+1))
                count=count+1
                count=dfs(row,column+1,count)
                Stack.pop()

    except IndexError:
        pass


    #checking (x+1, y-1) diagonal for child

    try:
        
        if  column!=0:

            if table[row+1][column-1]=='Y':

                if  str(row+1)+str(column+1) in visited :
                    pass

                else:
                    visited.append(str(row+1)+str(column-1))
                    Stack.append(str(row+1)+str(column-1))
                    count=count+1
                    count=dfs(row+1,column-1,count)
                    Stack.pop()

    except IndexError:
        pass


    #checking (x+1, y) down position for child

    try:

        if table[row+1][column]=='Y':

            if  str(row+1)+str(column) in visited :
                pass

            else:
                visited.append(str(row+1)+str(column))
                Stack.append(str(row+1)+str(column))
                count=count+1
                count=dfs(row+1,column,count)
                Stack.pop()

    except IndexError:
        pass


    #checking (x+1, y+1) diagonal position for child

    try:

        if table[row+1][column+1]=='Y':

            if  str(row+1)+str(column+1) in visited :
                pass
                
            else:
                visited.append(str(row+1)+str(column+1))
                Stack.append(str(row+1)+str(column+1))
                count=count+1
                count=dfs(row+1,column+1,count)
                Stack.pop()

    except IndexError:
        pass

    return  count


#reading file

file = open("input.txt")
lines_List = file.read().splitlines()
file.close()

table = [[]]

#storing values in the table

for i in range(len(lines_List)):

    for j in range(len(lines_List[0])):
        
        if (lines_List[i][j] != " "):

            table[i].append(lines_List[i][j])
    
    # removing extra row from table  
           
    if (i != len(lines_List)-1):

        table.append([])


visited=[]
Stack=[]
infected=[]

# traverse the table

for row   in  range(len(table)):

    for column   in  range(len(table[row])):

        count=0

        #if find Y in the table applying DFS

        if(table[row][column]=='Y'):

            if str(row)+str(column) in visited :
                pass

            else:
                visited.append(str(row)+str(column))
                Stack.append(str(row)+str(column))
                count=count+1
                count=dfs(row,column,count)
                infected.append(count)
                Stack.pop()
            

print("maximum infected area : ",max(infected))
