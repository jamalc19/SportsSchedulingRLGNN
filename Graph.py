from NodeAndEdge import Node,Edge
import xml.etree.ElementTree as ET
import os
import pickle

def splitmeetings(meetings):
    return [[int(team) for team in m.split(',')] for m in meetings.split(';')[:-1]]

class Graph:
    def __init__(self,teams,slots, hardconstraintcost):
        self.nodedict = {}
        self.nodemapping = {}
        self.teams=teams
        self.slots=slots
        self.hardconstraintcost=hardconstraintcost
        self.costconstant=0 #Some constraints are easier to model by giving the graph a cost and then subtracting the cost if nodes are selected.
        self.constraints={} #constraintid: [constrainttype, Max, penalty,edges,numconnectedselectednodes,addedweight]
                                    #constrainttype in ('node', 'edge') denotes if the constraint is a pick Max k out of n nodes or edges type of constraint
                                    #Max is the max number of nodes or edges that could be added before the constraint is violated
                                    #penalty is the cost for violating the constraint
                                    #edges is a list of edges related to this constraint
                                    #numconnectedselectednodes is a dict {nodeid: number of selected nodes connected to nodeid by an edge that's apart of this constraint}This only exists for constrainttype='edge'.
                                    #addedweight is a dict  (nodeid: addedweight to this node due to this constraint}
        self.constraintidcounter=0
        self.solution=set() #set of ids of the selected nodes
        self.forcedselections=set()
        self.solutionsize = (len(self.slots)*len(self.teams)/2)
        idnum = 0
        for s in self.slots:
            for hometeam in self.teams:
                for awayteam in self.teams:
                    if awayteam != hometeam:
                        self.nodedict[idnum] = Node(hometeam, awayteam, s, idnum)
                        self.nodemapping[(hometeam, awayteam, s)] = idnum
                        idnum += 1

    def deletenode(self, hometeam, awayteam, slot):
        idnum = self.nodemapping.get((hometeam, awayteam, slot))
        if idnum is None:
            return #node has already been deleted
        self.deletenodebyid(idnum)

    def deletenodebyid(self,idnum):
        node = self.nodedict.get(idnum)
        if node is None:
            return #node has already been deleted
        self.deletenodebyobject(node,idnum)

    def deletenodebyobject(self,node,idnum):
        del self.nodedict[idnum]
        del self.nodemapping[(node.hometeam, node.awayteam, node.slot)]
        node.delete()
        # TODO some complex constraints could be simplified because of node deletion. Won't effect correctness of reward or feasibility so leaving it for now.
        #ToDo select any nodes this forces. Example only one game left for a team in a slot. Or only one slot left for a matchup. To do this have to store sets for each of these and if set is singleton then it is forced
            #alternatively the RL model should pick this up. If RL is sequential then maybe this could be helpful

    def selectnode(self,nodeid):
        reward = self.computereward(nodeid)
        node = self.nodedict[nodeid]
        node.selected = True
        '''
        if True: #TODO remove. temporary to test Q learning
            self.solution.add(nodeid)
            return reward, len(self.solution) == self.solutionsize
        '''
        constraintids=set()
        for edge in node.edges_soft.keys(): #update non-complex soft constraints
            constraintids |= node.edges_soft[edge].constraintids
        for C_id in constraintids:
            self.updateconstraints(C_id, nodeid)

        constraintids = set()
        for edge in node.edges_hard_complex.keys(): #update complex hard constraints
            constraintids |= node.edges_hard_complex[edge].constraintids
        for C_id in constraintids:
            self.updateconstraints(C_id, nodeid)

        constraintids = set()
        for edge in node.edges_soft_complex.keys(): #update complex soft constraints
            constraintids |= node.edges_soft_complex[edge].constraintids
        for C_id in constraintids:
            self.updateconstraints(C_id, nodeid)

        for deletenodeid in set(node.edges_hard.keys()):  # delete all nodes connected to this one by a hard constraint
            self.deletenodebyid(deletenodeid)
        self.solution.add(nodeid)
        done=False
        feasible = True
        if len(self.solution)==self.solutionsize:
            done=True
        if len(self.nodedict) < self.solutionsize:  # RL agent reached an infeasible solution
            done = True
            reward-= self.hardconstraintcost
            feasible = False
        reward = max(reward,-self.hardconstraintcost)
        return reward, done, feasible

    def computereward(self,nodeid):
        node = self.nodedict[nodeid]
        cost=node.cost
        activenodes=self.solution
        for nodeid2 in (node.edges_soft.keys() & activenodes):
            cost+=node.edges_soft[nodeid2].weight
        for nodeid2 in (node.edges_hard.keys() & activenodes):
            cost+=node.edges_hard[nodeid2].weight
        return -cost

    def getActions(self):
        return self.nodedict.keys() - self.solution

    def updateconstraints(self,C_id, nodeid):
        #nodeid has been selected
        constraint = self.constraints[C_id]

        #NODE Constraint
        # decrement max of constraint by 1
        #if max>1
            #delete all edges going into selected node
            #update cost of other edges
        #if max=1 convert to a non complex constraint
            #delete all the complex edges
            #for every node part of the constraint that is not part of the solution (all remaining nodes), link it to all other nodes part of the constraint
        #if max=0 remove all arcs except for those from nodeid to unselected nodes


        #decrement max of constraint by 1
        if constraint[0]=='node':
            if constraint[1]==0:
                return
            constraint[1]= constraint[1]-1
            if constraint[1]==1:
                newconstraintid=self.constraintidcounter
                self.constraintidcounter+=1
                self.constraints[newconstraintid] = ['node', 1, constraint[2], set()]

            for edge in set(constraint[3]):
                if (edge.node1.id in self.nodedict) and (edge.node2.id in self.nodedict):#when nodes get deleted the list of edges for each constraint does not get updated.
                    if (edge.node1.id==nodeid) or (edge.node2.id==nodeid):
                        #selected node
                        if constraint[1] >= 1:  # max>=1
                            #delete constraint from this edge
                            constraint[3].remove(edge)
                            edge.deleteconstraint(C_id, constraint[2]/(constraint[1]+1))
                        #else max=0 and these are the only constraints we want to keep
                    else:
                        if constraint[1]>1:#max>1
                            # update cost of other edges
                            edge.addweight(constraint[2] / (constraint[1]) - constraint[2] / (constraint[1] + 1))
                        elif constraint[1]==1: #max=1
                            constraint[3].remove(edge)
                            edge.deleteconstraint(C_id, constraint[2] / (constraint[1] + 1))
                            self.addEdge(edge.node1,edge.node2,constraint[2],edge.hard,Complex=False, constraintid=newconstraintid)
                        else: #max=0
                            #delete constraint from this edge
                            constraint[3].remove(edge)
                            edge.deleteconstraint(C_id, constraint[2] / (constraint[1] + 1))
            if constraint[1] == 1:
                del self.constraints[C_id]
            return


        #EDGE Constraint
        #if break constraint then we're picking at most max out of k edges (no nodes)
        #if max =0, no updates are needed
        # decrement max of constraint by the amount of active edges create by selecting nodeid (could be 0-4)
        #for each edge
            #if it was just made active then remove. If max<0, the cost incurred for making max<0 was embedded into the node earlier
            #update cost of non-active edges
            #if max<=0 all non-active edges become non-complex constraints. Remove any costs embedded into nodes that have yet to be selected
            #if max>0:
                #update dict of {nodeid:num connected selected nodes}
                    #if this is  > max, then increment cost of node by weight*(numconnectedactivenodes-max)
        # if the decrement=0 only this last if block is needed because others will have no effect
        else:
            if constraint[1]==0: #if max already equals 0, edge constraints don't need any updating
                return
            maxdecrement = constraint[4].get(nodeid,0) #decrement max of constraint by the amount of active edges create by selecting nodeid (could be 0-4)
            constraint[1] = constraint[1] - maxdecrement
            for edge in set(constraint[3]):
                if (edge.node1.id in self.nodedict) and (edge.node2.id in self.nodedict):#when nodes get deleted the list of edges for each constraint does not get updated.
                    if (edge.node1.id==nodeid):
                        othernode = edge.node2
                    elif (edge.node2.id==nodeid):
                        othernode = edge.node1
                    else:
                        othernode=None
                    if othernode is not None:
                        # selected node is part of this edge
                        if othernode.selected: #this is an active edge because both nodes are selected
                            #remove
                            constraint[3].remove(edge)
                            edge.deleteconstraint(C_id, constraint[2] / (constraint[1] + maxdecrement + 1))
                            continue
                        else:
                            if constraint[1]>0:
                                constraint[4][othernode.id]= constraint[4].get(othernode.id,0)+1#update dict of {nodeid:num connected selected nodes}
                                if constraint[4][othernode.id]>constraint[1]: #if this is  > max, then increment cost of node by weight
                                    othernode.addcost(constraint[2])
                                    constraint[5][othernode.id] =  constraint[5].get(othernode.id,0) + constraint[2]
                    if maxdecrement==0:
                        continue#max hasn't changed so below updates will all have no effect
                    # if max<=0 all non-active edges become non-complex constraints. Remove any costs embedded into nodes that have yet to be selected
                    if constraint[1]<=0:
                        self.addEdge(edge.node1, edge.node2, constraint[2], edge.hard, Complex=False,
                                     constraintid=None)
                        edge.deleteconstraint(C_id, constraint[2] / (constraint[1] + maxdecrement+ 1))
                        for othernodeid in constraint[5]:
                            if othernodeid in self.nodedict:
                                self.nodedict[othernodeid].addcost(-constraint[5][othernodeid])
                    else:
                        #update cost of complex edges
                        edge.addweight(constraint[2] / (constraint[1]+1) - constraint[2] / (constraint[1] +maxdecrement + 1))

            if constraint[1]<=0:#if max<=0. We transformed it to a non-complex constraint and it will need no further updates
                del self.constraints[C_id]


    def addEdge(self, node1,node2,weight,hard=False,Complex=False, constraintid=None):
        #if edge already exists then increment cost. If complex then add complex id
        #else create new edge
        if node1.edges_hard.get(node2.id) is not None:
            # if a hard constraint already exists between these nodes then don't bother adding anything. Breaking this constraint already makes the problem infeasible
            #this will reduce size of graph and hopefully make edges more interpretable for struct2vec/RL algo
            return
        if hard:
            if Complex:
                edge = node1.edges_hard_complex.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                else:
                    edge= Edge(node1, node2, weight, hard, Complex, constraintid)
            else:
                #edge = node1.edges_hard.get(node2.id)
                #if edge is not None:
                    #edge.addweight(weight)
                    #pass
                #else:
                    #Edge(node1, node2, weight, hard, Complex, constraintid)
                edge=Edge(node1, node2, weight, hard, Complex, constraintid)
        else:
            if Complex:
                edge=node1.edges_soft_complex.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                else:
                    edge=Edge(node1, node2, weight, hard, Complex, constraintid)
            else:
                edge = node1.edges_soft.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                else:
                    edge=Edge(node1, node2, weight, hard, Complex, constraintid)

        if constraintid is not None:
            self.constraints[constraintid][3].add(edge)
            edge.constraintids.add(constraintid)

    def addonegameperweekconstraints(self):
        for slot in self.slots:
            for team in self.teams:
                nodes=[]
                for opponent in self.teams:
                    if team!=opponent:
                        nodeid= self.nodemapping.get((team, opponent, slot))
                        if nodeid is not None:
                            nodes.append(self.nodedict[nodeid])
                        nodeid= self.nodemapping.get((opponent, team, slot))
                        if nodeid is not None:
                            nodes.append(self.nodedict[nodeid])
                for i in range(len(nodes)-1):
                    for j in range(i+1,len(nodes)):
                        self.addEdge(nodes[i],nodes[j],weight=self.hardconstraintcost,hard=True)

    def adduniquematchupconstraints(self):
        for hometeam in self.teams:
            for awayteam in self.teams:
                if hometeam != awayteam:
                    nodes = []
                    for slot in self.slots:
                        nodeid = self.nodemapping.get((hometeam, awayteam, slot))
                        if nodeid is not None:
                            nodes.append(self.nodedict[nodeid])
                    for i in range(len(nodes) - 1):
                        for j in range(i + 1, len(nodes)):
                            self.addEdge(nodes[i], nodes[j], weight=self.hardconstraintcost, hard=True)

    def addphasedconstraints(self):
        midpoint = int(len(self.slots)/2)
        for i in range(len(self.teams)-1):
            for j in range(i+1,len(self.teams)):
                #firsthalf
                nodelist1=[] #list of games where team i is home against team j
                nodelist2=[] #list of games where team j is home against team i
                for s in range(midpoint):
                    nodeid = self.nodemapping.get((self.teams[i], self.teams[j], s))
                    if nodeid is not None:
                        nodelist1.append(self.nodedict[nodeid])
                    nodeid = self.nodemapping.get((self.teams[j], self.teams[i], s))
                    if nodeid is not None:
                        nodelist2.append(self.nodedict[nodeid])
                for n1 in nodelist1:
                    for n2 in nodelist2:
                        self.addEdge(n1, n2, weight=self.hardconstraintcost, hard=True)

                #second half
                nodelist1=[] #list of games where team i is home against team j
                nodelist2=[] #list of games where team j is home against team i
                for s in range(midpoint,len(self.slots)):
                    nodeid = self.nodemapping.get((self.teams[i], self.teams[j], s))
                    if nodeid is not None:
                        nodelist1.append(self.nodedict[nodeid])
                    nodeid = self.nodemapping.get((self.teams[j], self.teams[i], s))
                    if nodeid is not None:
                        nodelist2.append(self.nodedict[nodeid])
                for n1 in nodelist1:
                    for n2 in nodelist2:
                        self.addEdge(n1, n2, weight=self.hardconstraintcost, hard=True)

    def addCA1(self, C):
        penalty=int(C.get('penalty'))
        team = int(C.get('teams'))
        mode = C.get('mode')
        hard= C.get('type')=='HARD'
        if hard:
            penalty=self.hardconstraintcost
        slots= [int(s) for s in C.get('slots').split(';')]
        Max = int(C.get('max'))
        if (Max==0): #disallowed games
            if hard:
                for s in slots:
                    if mode=='H':
                        # remove all home games in this slot
                        for awayteam in self.teams:
                            if team!=awayteam:
                                self.deletenode(team, awayteam, s)
                    else:
                        # remove all away games in this slot
                        for hometeam in self.teams:
                            if team != hometeam:
                                self.deletenode(hometeam, team, s)
            else: #soft
                for s in slots:
                    if mode=='H':
                        # add cost to all home games in this slot
                        for awayteam in self.teams:
                            if team!=awayteam:
                                nodeid = self.nodemapping.get((team, awayteam, s))
                                if nodeid is not None:
                                    self.nodedict[nodeid].addcost(penalty)
                    else:
                        # add cost to all away games in this slot
                        for hometeam in self.teams:
                            if team != hometeam:
                                nodeid = self.nodemapping.get((hometeam, team, s))
                                if nodeid is not None:
                                    self.nodedict[nodeid].addcost(penalty)
        else:
            # We will add constraint between every game in a slot to every game in another slot
            #Get all relevant nodes first
            slotgames=[] #list of lists. Inner list has all the relevant games of a team in a given slot
            for slot in slots:
                games=[]
                if mode == 'H':
                    # add cost to all home games in this slot
                    for awayteam in self.teams:
                        if team != awayteam:
                            nodeid = self.nodemapping.get((team, awayteam, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                else:
                    # add cost to all away games in this slot
                    for hometeam in self.teams:
                        if team != hometeam:
                            nodeid = self.nodemapping.get((hometeam, team, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                slotgames.append(games)
            #determine if complex or not
            constraintid = self.constraintidcounter
            self.constraints[constraintid] = ['node',Max,penalty,set()]
            self.constraintidcounter += 1
            if Max==1:
                Complex=False
            else: #if max>1 then it is a complex constraint
                Complex=True
                penalty = penalty/Max #TODO tune how penalty is split between complex arcs. Incorporate len(slots)??
            #add all the edges
            for i in range(len(slots)-1):
                    for j in range(i+1,len(slots)):
                        for node_i in slotgames[i]:
                            for node_j in slotgames[j]:
                                self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex,constraintid=constraintid)

    def addCA2(self, C):
        penalty=int(C.get('penalty'))
        team = int(C.get('teams1'))
        opponents = [int(o) for o in C.get('teams2').split(';')]
        mode = C.get('mode1')
        hard= C.get('type')=='HARD'
        hard=False #TODO Remove this if using ITC instances
        if hard:
            penalty=self.hardconstraintcost
        slots= [int(s) for s in C.get('slots').split(';')]
        Max = int(C.get('max'))
        Min = int(C.get('min'))
        if (Max == 0):
            if hard:
                for s in slots:
                    if 'H' in mode:
                        # remove all home games in this slot
                        for awayteam in opponents:
                            self.deletenode(team, awayteam, s)
                    if 'A' in mode:
                        # remove all away games in this slot
                        for hometeam in opponents:
                            self.deletenode(hometeam, team, s)
            else:
                for s in slots:
                    if 'H' in mode:
                        # remove all home games in this slot
                        for awayteam in opponents:
                            nodeid = self.nodemapping.get((team, awayteam, s))
                            if nodeid is not None:
                                self.nodedict[nodeid].addcost(penalty)
                    if 'A' in mode:
                        # remove all away games in this slot
                        for hometeam in opponents:
                            nodeid = self.nodemapping.get((hometeam, team, s))
                            if nodeid is not None:
                                self.nodedict[nodeid].addcost(penalty)
        else:
            # We will add constraint between every game in a slot to every game in another slot
            # Get all relevant nodes first
            slotgames = []  # list of lists. Inner list has all the relevant games of a team in a given slot
            for slot in slots:
                games = []
                if 'H' in mode:
                    # add cost to all home games in this slot against one of opponents
                    for awayteam in opponents:
                        nodeid = self.nodemapping.get((team, awayteam, slot))
                        if nodeid is not None:
                            games.append(self.nodedict[nodeid])
                if 'A' in mode:
                    # add cost to all away games in this slot against one of opponents
                    for hometeam in opponents:
                        nodeid = self.nodemapping.get((hometeam, team, slot))
                        if nodeid is not None:
                            games.append(self.nodedict[nodeid])
                slotgames.append(games)

            constraintid = self.constraintidcounter
            self.constraints[constraintid]=['node',Max,penalty,set()]
            self.constraintidcounter += 1
            # determine if complex or not
            if Max == 1:
                Complex = False
            else:  # if max>1 then it is a complex constraint
                Complex = True
                penalty = penalty / Max  # TODO tune how penalty is split between complex arcs. Incorporate len(slots)??
            # add all the edges
            for i in range(len(slots) - 1):
                for j in range(i + 1, len(slots)):
                    for node_i in slotgames[i]:
                        for node_j in slotgames[j]:
                            self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex, constraintid=constraintid)


    def addCA3(self, C):
        # all constraints in our data are either 3,2,0 or 4,2,0
        Max = int(C.get('max'))
        Min = int(C.get('min'))
        intp = int(C.get('intp'))
        mode = C.get('mode1')
        penalty = int(C.get('penalty'))
        teams1 = [int(o) for o in C.get('teams1').split(';')]
        teams2 = [int(o) for o in C.get('teams2').split(';')]

        hard = C.get('type') == 'HARD'
        if hard:
            penalty=self.hardconstraintcost
        #In our data: max=2, min=0, intp=3 or 4
        #if intp=4, then teams1 is a singleton and teams2 is a subset. could be H, A or HA

        #separate constraint for each team and each window of length intp
        #create all arcs.
        # as arcs become active then decrement Max. If Max=1 then convert all edges into non-complex constraints
        Complex = True

        #all constraints in our dataset our complex because Max>1
        originalpenalty=penalty
        penalty = penalty / Max
        for team in teams1:
            for slotend in range(intp, len(self.slots)+1):
                #new constraint for every team and every window of slots of length intp
                constraintid = self.constraintidcounter
                self.constraints[constraintid] =['node',Max,originalpenalty,set()]
                self.constraintidcounter += 1
                slotgames=[]
                for slot in range(slotend-intp,slotend):
                    games=[]
                    if 'H' in mode:
                        for awayteam in teams2:
                            nodeid = self.nodemapping.get((team, awayteam, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                    if 'A' in mode:
                        for hometeam in teams2:
                            nodeid = self.nodemapping.get((hometeam, team, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                    slotgames.append(games)
                # add all the edges
                for i in range(len(slotgames) - 1):
                    for j in range(i + 1, len(slotgames)):
                        for node_i in slotgames[i]:
                            for node_j in slotgames[j]:
                                self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex, constraintid=constraintid)


    def addCA4(self, C):
        #in our dataset Max>0, Min=0

        #like CA2 but teams1 is treated as one single entity
        #GLOBAl is the same
        #Every is repeated constraint in each slot in slots
        penalty = int(C.get('penalty'))
        teams = [int(t) for t in C.get('teams1').split(';')]
        opponents = [int(o) for o in C.get('teams2').split(';')]
        mode = C.get('mode1')
        mode2 = C.get('mode2')
        hard = C.get('type') == 'HARD'
        if hard:
            penalty=self.hardconstraintcost
        slots = [int(s) for s in C.get('slots').split(';')]
        Max = int(C.get('max'))
        if mode2=='GLOBAL':
            # Get all relevant nodes first
            games = []  # list of lists. Inner list has all the relevant games of a team in a given slot
            for slot in slots:
                if 'H' in mode:
                    # add cost to all home games between team and opponent
                    for hometeam in teams:
                        for awayteam in opponents:
                            nodeid = self.nodemapping.get((hometeam, awayteam, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                if 'A' in mode:
                    # add cost to all away games between team and opponent
                    for awayteam in teams:
                        for hometeam in opponents:
                            nodeid = self.nodemapping.get((hometeam, awayteam, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
            constraintid = self.constraintidcounter
            self.constraints[constraintid] =['node',Max,penalty,set()]
            self.constraintidcounter += 1
            # determine if complex or not
            if Max == 1:
                Complex = False
            else:  # if max>1 then it is a complex constraint
                Complex = True
                penalty = penalty / Max  # TODO tune how penalty is split between complex arcs. Incorporate len(slots)??
            # add all the edges
            for i in range(len(games) - 1):
                for j in range(i+1, len(games)):
                    node_i = games[i]
                    node_j = games[j]
                    self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex, constraintid=constraintid)
        else:#mode2=='EVERY
            # separate constraint for each slot
            originalpenalty=penalty
            if Max!=1: #will be a complex constraint
                penalty = penalty / Max  # TODO tune how penalty is split between complex arcs.
            for slot in slots:
                constraintid = self.constraintidcounter
                self.constraints[constraintid] = ['node',Max,originalpenalty,set()]
                self.constraintidcounter += 1
                # determine if complex or not
                if Max == 1:
                    Complex = False
                else:  # if max>1 then it is a complex constraint
                    Complex = True
                games = []
                if 'H' in mode:
                    # add cost to all home games between team and opponent
                    for hometeam in teams:
                        for awayteam in opponents:
                            nodeid = self.nodemapping.get((hometeam, awayteam, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                if 'A' in mode:
                    # add cost to all away games between team and opponent
                    for awayteam in teams:
                        for hometeam in opponents:
                            nodeid = self.nodemapping.get((hometeam, awayteam, slot))
                            if nodeid is not None:
                                games.append(self.nodedict[nodeid])
                # add all the edges
                for i in range(len(games) - 1):
                    for j in range(i+1, len(games)):
                        node_i = games[i]
                        node_j = games[j]
                        self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex, constraintid=constraintid)

    def addGA1(self, C):
        penalty=int(C.get('penalty'))
        meetings = splitmeetings(C.get('meetings'))
        hard= C.get('type')=='HARD'
        if hard:
            penalty=self.hardconstraintcost
        slots= [int(s) for s in C.get('slots').split(';')]
        Max = int(C.get('max'))
        Min = int(C.get('min'))
        if Max==0:
            #games are forbidden. remove if hard or add cost to node if soft
            for s in slots:
                for m in meetings:
                    if hard:
                        self.deletenode(m[0],m[1],s)
                    else:
                        nodeid = self.nodemapping.get((m[0],m[1],s))
                        if nodeid is not None:
                            self.nodedict[nodeid].addcost(penalty)
        elif (Min==Max)and (Min==len(meetings)):
            #meetings must be in these slots so can't be elsewhere
            if hard:
                #Delete those matchups from all other slots
                for m in meetings:
                    for s in self.slots:
                        if not s in slots:
                            self.deletenode(m[0], m[1], s)
            else:
                # give any solution to the graph a cost, and then remove the cost if the node is selected
                for m in meetings:
                    for s in self.slots:
                        if not s in slots:
                            nodeid = self.nodemapping.get((m[0],m[1],s))
                            if nodeid is not None:
                                self.nodedict[nodeid].addcost(-penalty)
                                self.costconstant+=penalty

            if (len(slots)==1):
                # these games are fixed
                if hard:
                    s=slots[0]
                    for m in meetings:
                        #add game to solution
                        nodeid=self.nodemapping[(m[0],m[1],s)]
                        self.forcedselections.add(nodeid)
                else:
                    s = slots[0]
                    for m in meetings:
                        #penalize graph if game not added game to solution
                        nodeid=self.nodemapping.get((m[0],m[1],s))
                        self.costconstant += penalty
                        if nodeid is not None:
                            self.nodedict[nodeid].addcost(-penalty)
            else:
                #if a team is apart of every matchup then that team cannot play a different matchup in slots
                #remove nodes if constraint is hard. If soft the cost is already taken care of by first if else statement
                if hard:
                    teams=set(meetings[0])
                    for m in meetings[1:]:
                        teams-=set(m)
                    if len(teams)>0:
                        for team in teams:
                            for s in slots:
                                for opponent in self.teams:
                                    if opponent!=team:
                                        if not [opponent,team] in meetings:
                                            self.deletenode(opponent,team,s)
                                        if not [team, opponent] in meetings:
                                            self.deletenode(team, opponent, s)

        elif Min<Max:
            #if (Max < len(meetings)) and (Min>0):
                #this never occurs in our data
            if (Min>0): #max= len(meetings)
                # if not in this block then min=0, max < len(meetings)
                #we need to select at least min out of max games
                #equivalent to not selecting more than (Max-Min) out of meetings in *other* slots
                Max=Max-Min
                slots = [s for s in self.slots if s not in slots]
                #now the constraint is written as if it was a min=0, max < len(meetings) GA1 constraint

            # we cannot select more than max out of len(meetings) games in slots
            constraintid = self.constraintidcounter
            self.constraints[constraintid] = ['node',Max,penalty,set()]
            self.constraintidcounter += 1
            if Max==1:
                Complex=False
            else:
                Complex=True
                penalty = penalty / Max  # TODO tune how penalty is split between complex arcs

            meetinggames=[]
            for meeting in meetings:
                games=[]
                for slot in slots:
                    nodeid = self.nodemapping.get((meeting[0], meeting[1], slot))
                    if nodeid is not None:
                        games.append(self.nodedict[nodeid])
                meetinggames.append(games)

            for i in range(len(meetinggames) - 1):
                for j in range(i + 1, len(meetinggames)):
                    for node_i in meetinggames[i]:
                        for node_j in meetinggames[j]:
                            self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex,
                                         constraintid=constraintid)
        #elif (Min==Max) and Max<len(meetings):
            #this never occurs in our data



    def addBR1(self, C):
        intp = int(C.get('intp'))
        slots = [int(s) for s in C.get('slots').split(';')]
        teams = [int(t) for t in C.get('teams').split(';')]
        mode = C.get('mode2')
        penalty=int(C.get('penalty'))
        hard= C.get('type')=='HARD'
        if hard:
            penalty=self.hardconstraintcost
        #intp is 0,1, or 2 in our data
        for team in teams:
            if intp>0:
                Complex=True
                constraintid=self.constraintidcounter
                self.constraintidcounter+=1
                self.constraints[constraintid]=['edge',intp,penalty,set(),{},{}]
                penalty = penalty/(intp+1)#todo update complex penalty logic
                #intp. on selection of constraint decrement intp. if intp=0 then create non-complex constraint
            else:
                Complex=False
                constraintid=None

            for opponent1 in self.teams:
                if opponent1!=team:
                    for opponent2 in self.teams:
                        if (opponent2!=opponent1) and (opponent2!=team):
                            for slot in slots:
                                if 'H' in mode:
                                    nodeid1 = self.nodemapping.get((team, opponent1, slot))
                                    nodeid2 = self.nodemapping.get((team, opponent2, slot-1))
                                    if (nodeid1 is not None) and (nodeid2 is not None):
                                        self.addEdge(self.nodedict[nodeid1],self.nodedict[nodeid2],penalty,hard,Complex,constraintid)
                                    nodeid1 = self.nodemapping.get((team, opponent2, slot))
                                    nodeid2 = self.nodemapping.get((team, opponent1, slot-1))
                                    if (nodeid1 is not None) and (nodeid2 is not None):
                                        self.addEdge(self.nodedict[nodeid1],self.nodedict[nodeid2],penalty,hard,Complex,constraintid)
                                if 'A' in mode:
                                    nodeid1 = self.nodemapping.get((opponent1, team, slot))
                                    nodeid2 = self.nodemapping.get((opponent2, team, slot - 1))
                                    if (nodeid1 is not None) and (nodeid2 is not None):
                                        self.addEdge(self.nodedict[nodeid1], self.nodedict[nodeid2], penalty, hard,Complex,constraintid)
                                    nodeid1 = self.nodemapping.get((opponent2, team, slot))
                                    nodeid2 = self.nodemapping.get((opponent1, team, slot - 1))
                                    if (nodeid1 is not None) and (nodeid2 is not None):
                                        self.addEdge(self.nodedict[nodeid1], self.nodedict[nodeid2], penalty, hard,Complex,constraintid)


    def addBR2(self, C):
        teams = [int(t) for t in C.get('teams').split(';')]
        slots= [int(s) for s in C.get('slots').split(';')]
        intp = int(C.get('intp'))
        penalty=int(C.get('penalty'))
        hard= C.get('type')=='HARD'
        if hard:
            penalty=self.hardconstraintcost
        #this constraint only exists in our data across the whole season
        if intp > 0:
            Complex = True
            constraintid = self.constraintidcounter
            self.constraintidcounter += 1
            self.constraints[constraintid] = ['edge',intp,penalty,set(),{},{}]
            penalty = penalty / (intp + 1)  # todo update complex penalty logic
        else:
            Complex = False
            constraintid = None
        for team in teams:
            for opponent1 in teams:
                if opponent1!=team:
                    for opponent2 in teams:
                        if (opponent2!=opponent1) and (opponent2!=team):
                            for slot in self.slots[1:]:
                                #home breaks
                                nodeid1 = self.nodemapping.get((team, opponent1, slot))
                                nodeid2 = self.nodemapping.get((team, opponent2, slot-1))
                                if (nodeid1 is not None) and (nodeid2 is not None):
                                    self.addEdge(self.nodedict[nodeid1],self.nodedict[nodeid2],penalty,hard,Complex,constraintid)
                                nodeid1 = self.nodemapping.get((team, opponent2, slot))
                                nodeid2 = self.nodemapping.get((team, opponent1, slot-1))
                                if (nodeid1 is not None) and (nodeid2 is not None):
                                    self.addEdge(self.nodedict[nodeid1],self.nodedict[nodeid2],penalty,hard,Complex,constraintid)
                                #away breaks
                                nodeid1 = self.nodemapping.get((opponent1, team, slot))
                                nodeid2 = self.nodemapping.get((opponent2, team, slot - 1))
                                if (nodeid1 is not None) and (nodeid2 is not None):
                                    self.addEdge(self.nodedict[nodeid1], self.nodedict[nodeid2], penalty, hard,Complex,constraintid)
                                nodeid1 = self.nodemapping.get((opponent2, team, slot))
                                nodeid2 = self.nodemapping.get((opponent1, team, slot - 1))
                                if (nodeid1 is not None) and (nodeid2 is not None):
                                    self.addEdge(self.nodedict[nodeid1], self.nodedict[nodeid2], penalty, hard,Complex,constraintid)


    def addFA2(self, C):
        # TODO
        #always a soft constraint for every team for every slot with intp=2
        intp = int(C.get('intp'))

        #could limit homebreaks and hope that helps with FA2.
        #always soft so could just ignore. Also only exists in about half the instances
        pass


    def addSE1(self, C):
        teams = [int(t) for t in C.get('teams').split(';')]
        penalty=int(C.get('penalty'))
        hard= C.get('type')=='HARD'
        if hard:
            penalty = self.hardconstraintcost
        Min = int(C.get('min'))
        for s1 in range(len(self.slots)):
            for s2 in range(s1+1, min(s1+Min+2,len(self.slots))):
                for i in range(len(teams)-1):
                    team1=teams[i]
                    for j in range(i+1,len(teams)):
                        team2=teams[j]
                        #first direction
                        nodeid1= self.nodemapping.get((team1,team2,s1))
                        nodeid2 = self.nodemapping.get((team2, team1, s2))
                        if (nodeid1 is not None) and (nodeid2 is not None):
                            self.addEdge(self.nodedict[nodeid1], self.nodedict[nodeid2], penalty, hard)
                        #second direction
                        nodeid1= self.nodemapping.get((team2,team1,s1))
                        nodeid2 = self.nodemapping.get((team1, team2, s2))
                        if (nodeid1 is not None) and (nodeid2 is not None):
                            self.addEdge(self.nodedict[nodeid1], self.nodedict[nodeid2], penalty, hard)



def creategraph(path, hardconstraintcost=10000):
    tree = ET.parse(path)
    root=tree.getroot()
    phased= root.find('Structure').find('Format').find('gameMode')=='P'
    resources = root.find('Resources')
    teams = [int(child.attrib['id']) for child in resources.find('Teams')]
    slots = [int(child.attrib['id']) for child in resources.find('Slots')]
    G=Graph(teams,slots,hardconstraintcost)

    constraints = root.find('Constraints')
    capacityconstraints = constraints.find('CapacityConstraints')
    gameconstraints = constraints.find('GameConstraints')
    breakconstraints = constraints.find('BreakConstraints')
    fairnessconstraints = constraints.find('FairnessConstraints')
    separationconstraints =constraints.find('SeparationConstraints')

    # Doing node eliminating constraints first doesn't increase efficiency by much, so add constraints in readable order
    #add standard constraints of the double round robin format
    G.addonegameperweekconstraints()
    G.adduniquematchupconstraints()
    if phased:
        G.addphasedconstraints()

    #add GA1 constraints some of which are node eliminating
    if gameconstraints:
        for GA1 in gameconstraints:
            G.addGA1(GA1.attrib)

    #add CA constraints. Some of CA1 and CA2 are node eliminating.
    if capacityconstraints:
        for C in capacityconstraints:
            if C.tag=='CA1':
                G.addCA1(C)
            elif C.tag=='CA2':
                G.addCA2(C)
            elif C.tag=='CA3':
                G.addCA3(C)
            elif C.tag=='CA4':
                G.addCA4(C)

    #add break constraints
    if breakconstraints:
        for B in breakconstraints:
            if B.tag=='BR1':
                G.addBR1( B.attrib)
            elif B.tag=='BR2':
                G.addBR2( B.attrib)
            else:
                print('unknown constraint',B)
    #add fairness constraints
    if fairnessconstraints:
        for FA2 in fairnessconstraints:
            G.addFA2(FA2.attrib)
    #add separation constraints
    if separationconstraints:
        for SE1 in separationconstraints:
            G.addSE1(SE1.attrib)
    for nodeid in G.forcedselections:
        G.selectnode(nodeid)
    return G


if __name__=='__main__':
    #to avoid max recursion error when saving pickle
    import sys
    max_rec = 0x100000
    sys.setrecursionlimit(max_rec)
    # files = ['ITC2021_Test1.xml','ITC2021_Test2.xml','ITC2021_Test3.xml','ITC2021_Test4.xml']
    # for file in files:
    #     g = creategraph('Instances/' + file,hardconstraintcost=1)
    #     for node in g.nodedict.values():
    #         node.cost=0#TODO for hard constraint testing only
    for file in os.listdir('GenInstances/'):
        g = creategraph('GenInstances/'+file, hardconstraintcost=10000)
        pickle.dump(g, open('PreprocessedInstances/' + file.replace('xml','pkl'),'wb'))

    #print(len(g.teams),len(g.nodedict), len(g.nodedict)/(2*len(g.teams)*(len(g.teams)-1)**2)) #max num of nodes is 2*n*(n-1)^2

    
