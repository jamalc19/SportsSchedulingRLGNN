from NodeAndEdge import Node,Edge
import xml.etree.ElementTree as ET
import os

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
        self.complexconstraints={} #complexid: list of edges related to this complex constraint
        self.complexidcounter=0
        self.solution=set() #set of ids of the selected nodes
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
        for complexconstraint in self.nodedict[idnum].edges_hard_complex:
            #TODO
            pass
        for complexconstraint in self.nodedict[idnum].edges_soft_complex:
            #TODO
            pass
        self.nodedict[idnum].delete()
        del self.nodedict[idnum]
        del self.nodemapping[(hometeam, awayteam, slot)]

    def deletenodebyid(self,idnum):
        #ToDo link all the delete node methods together
        node = self.nodedict.get(idnum)
        if node is None:
            return #node has already been deleted
        del self.nodemapping[(node.hometeam, node.awayteam, node.slot)]
        node.delete()
        del self.nodedict[idnum]
        #ToDo select any nodes this forces. Example only one game left for a team in a slot. Or only one slot left for a matchup
            #alternatively the RL model should pick this up. If RL is sequential then maybe this is needed

    def deletenodebyobject(self,node):
        del self.nodedict[node.idnum]
        del self.nodemapping[(node.hometeam, node.awayteam, node.slot)]
        node.delete()

    def selectnode(self,nodeid):
        reward = self.computereward(nodeid)
        affectedcomplexconstraintids,nodestodelete = self.nodedict[nodeid].select()
        self.solution.add(nodeid)
        for nodeid in nodestodelete:
            self.deletenodebyid(nodeid)
        for C_id in affectedcomplexconstraintids:
            self.updatecomplexconstraint(C_id)
        return reward

    def computereward(self,nodeid):
        node = self.nodedict[nodeid]
        cost=node.cost
        activenodes=self.solution
        for nodeid2 in (node.edges_soft.keys() & activenodes):
            cost+=node.edges_soft[nodeid2].weight
        for nodeid2 in (node.edges_hard.keys() & activenodes):
            cost+=node.edges_soft[nodeid2].weight
        return -cost

    def getActions(self):
        return self.nodedict.keys() - self.solution

    def updatecomplexconstraint(self,C_id):
        pass
        #TODO


    def addEdge(self, node1,node2,weight,hard=False,Complex=False, complexid=None):
        #if edge already exists then increment cost. If complex then add complex id
        #else create new edge
        if hard:
            if Complex:
                edge = node1.edges_hard_complex.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                    edge.complexids.append(complexid)
                else:
                    edge= Edge(node1, node2, weight, hard, Complex, complexid)
                self.complexconstraints[complexid].append(edge)
            else:
                edge = node1.edges_hard.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                else:
                    Edge(node1, node2, weight, hard, Complex, complexid)
        else:
            if Complex:
                edge=node1.edges_soft_complex.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                    edge.complexids.append(complexid)
                else:
                    Edge(node1, node2, weight, hard, Complex, complexid)
                self.complexconstraints[complexid].append(edge)
            else:
                edge = node1.edges_soft.get(node2.id)
                if edge is not None:
                    edge.addweight(weight)
                else:
                    Edge(node1, node2, weight, hard, Complex, complexid)

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
                        if nodes[i].edges_hard.get(nodes[j]) is not None:  #avoid adding the same constraint twice
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
            if Max==1:
                Complex=False
                complexid=None
            else: #if max>1 then it is a complex constraint
                Complex=True
                complexid=self.complexidcounter
                self.complexconstraints[complexid]=[]
                self.complexidcounter+=1
                penalty = penalty/Max #TODO tune how penalty is split between complex arcs. Incorporate len(slots)??
            #add all the edges
            for i in range(len(slots)-1):
                    for j in range(i+1,len(slots)):
                        for node_i in slotgames[i]:
                            for node_j in slotgames[j]:
                                self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex,complexid=complexid)

    def addCA2(self, C):
        penalty=int(C.get('penalty'))
        team = int(C.get('teams1'))
        opponents = [int(o) for o in C.get('teams2').split(';')]
        mode = C.get('mode1')
        hard= C.get('type')=='HARD'
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
            # determine if complex or not
            if Max == 1:
                Complex = False
                complexid = None
            else:  # if max>1 then it is a complex constraint
                Complex = True
                complexid = self.complexidcounter
                self.complexconstraints[complexid]=[]
                self.complexidcounter += 1
                penalty = penalty / Max  # TODO tune how penalty is split between complex arcs. Incorporate len(slots)??
            # add all the edges
            for i in range(len(slots) - 1):
                for j in range(i + 1, len(slots)):
                    for node_i in slotgames[i]:
                        for node_j in slotgames[j]:
                            self.addEdge(node_i, node_j, weight=penalty, hard=hard, Complex=Complex, complexid=complexid)


    def addCA3(self, C):
        # TODO
        Max = int(C.get('max'))
        Min = int(C.get('min'))
        intp = int(C.get('intp'))
        teams1 = [int(o) for o in C.get('teams1').split(';')]
        teams2 = [int(o) for o in C.get('teams2').split(';')]

        #all constraints in our data are either 3,2,0 or 4,2,0

        #for 4,2,0. add constant cost to graph. add c


    def addCA4(self, C):
        # TODO
        #like CA2 but teams1 is treated as one single entity
        #GLOBAl is the same
        #Every is repeated constraint in each slot in slots
        pass

    def addGA1(self, C):
        penalty=int(C.get('penalty'))
        meetings = splitmeetings(C.get('meetings'))
        hard= C.get('type')=='HARD'
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
                        self.selectnode(nodeid)
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
                #remove nodes if constraint is hard. If soft the cost is already taken care of ny first if esle statement
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
            if (Min>0):
                #todo
                pass
                #we need to select at least min out of max games
                #encourage selection of meetings

                #update: As meetings get deleted, this constraint could turn into a forced game
                #as nodes get selected, this constraint could get deleted

            else:
                #todo
                pass
                #we cannot select more than max out of len(meetings) games
                #discourage selection of meetings

                #update: As meetings get deleted, this constraint get could deleted
                #as meetings get selected, this could turn into a forbidden game

        #elif (Min==Max) and Max<len(meetings):
            #this never occurs in our data



    def addBR1(self, C):
        # TODO
        intp = int(C.get('intp'))
        slots = [int(s) for s in C.get('slots').split(';')]
        #intp is 0,1, or 2
        if intp==0:
            #forbid breaks
            pass
        else:
            #complex constraint for each break
            #once a break is selected then forbid other breaks
            pass


    def addBR2(self, C):
        # TODO
        teams = [int(t) for t in C.get('teams').split(';')]
        slots= [int(s) for s in C.get('slots').split(';')]
        if not ((len(slots)==len(self.slots)) &(len(teams)==len(self.teams))):
            print(len(slots),len(self.slots),len(teams),len(self.teams))
        #this constraint only exists across the whole season
        pass


    def addFA2(self, C):
        # TODO
        #always a soft constraint for every team for every slot with intp=2
        intp = int(C.get('intp'))

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


def creategraph(path, hardconstraintcost=100):
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
    for GA1 in gameconstraints:
        G.addGA1(GA1.attrib)

    #add CA constraints. Some of CA1 and CA2 are node eliminating.
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
    for B in breakconstraints:
        if B.tag=='BR1':
            G.addBR1( B.attrib)
        elif B.tag=='BR2':
            G.addBR2( B.attrib)
        else:
            print('unknown constraint',B)
    #add fairness constraints
    for FA2 in fairnessconstraints:
        G.addFA2(FA2.attrib)
    #add separation constraints
    for SE1 in separationconstraints:
        G.addSE1(SE1.attrib)
    return G



if __name__=='__main__':
    graphs=[creategraph('Instances/'+file) for file in os.listdir('Instances/')]
    for g in graphs:
        print(len(g.teams),len(g.nodedict), len(g.nodedict)/(2*len(g.teams)*(len(g.teams)-1)**2)) #max num of nodes is 2*n*(n-1)^2
