from NodeAndEdge import Node,Edge
import xml.etree.ElementTree as ET
import os

def splitmeetings(meetings):
    return [m.split(',') for m in meetings.split(';')[:-1]]

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
        self.solution=[] #list of ids of the selected nodes

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
        node = self.nodedict.get(idnum)
        if node is None:
            return #node has already been deleted
        del self.nodemapping[(node.hometeam, node.awayteam, node.slot)]
        node.delete()
        del self.nodedict[idnum]

    def deletenodebyobject(self,node):
        del self.nodedict[node.idnum]
        del self.nodemapping[(node.hometeam, node.awayteam, node.slot)]
        node.delete()

    def selectnode(self,nodeid):
        affectedcomplexconstraintids = self.nodedict[nodeid].select()
        self.solution.append(nodeid)
        for C_id in affectedcomplexconstraintids:
            self.updatecomplexconstraint(C_id)
        return #TODO return reward

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
                    Edge(node1, node2, weight, hard, Complex, complexid)
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
        team = C.get('teams')
        mode = C.get('mode')
        hard= C.get('type')=='HARD'
        if hard:
            penalty=self.hardconstraintcost
        slots= C.get('slots').split(';')
        Max = int(C.get('max'))
        if (Max==0):
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

        if Max==1:# add constraint between every game in a slot to every game in another slot
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
            for i in range(len(slots)-1):
                    for j in range(i,len(slots)):
                        for node_i in slotgames[i]:
                            for node_j in slotgames[j]:
                                self.addEdge(node_i, node_j, weight=penalty, hard=hard)
        else:#max>=2
            # add complex constraint between every game in a slot to every game in another slot
            pass

            #if max>=2
            #add complex constraints
            #TODO

    def addCA2(self, C):
        penalty=int(C.get('penalty'))
        team = C.get('teams1')
        opponents = C.get('teams2')
        mode = C.get('mode1')
        hard= C.get('type')=='HARD'
        slots= C.get('slots').split(';')
        Max = int(C.get('max'))
        Min = int(C.get('min'))
        if (Max == 0) and (hard):
            for s in slots:
                if 'H' in mode:
                    # remove all home games in this slot
                    for awayteam in opponents:
                        self.deletenode(team, awayteam, s)
                if 'A' in mode:
                    # remove all away games in this slot
                    for hometeam in opponents:
                        self.deletenode(hometeam, team, s)
        #TODO


    def addCA3(self, C):
        # TODO
        pass


    def addCA4(self, C):
        # TODO
        pass

    def addGA1(self, C):
        penalty=int(C.get('penalty'))
        meetings = splitmeetings(C.get('meetings'))
        hard= C.get('type')=='HARD'
        slots= C.get('slots').split(';')
        Max = int(C.get('max'))
        Min = int(C.get('min'))
        if Max==0:
            for s in slots:
                for m in meetings:
                    if hard:
                        self.deletenode(m[0],m[1],s)
                    else:
                        nodeid = self.nodemapping.get((m[0],m[1],s))
                        if nodeid is not None:
                            self.nodedict[nodeid].addcost(penalty)
        if (Min==Max)and (Min==len(meetings)):
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
                        nodeid = self.nodemapping.get((m[0],m[1],s))
                        if nodeid is not None:
                            self.nodedict[nodeid].addcost(-penalty)
                            self.costconstant+=penalty
            if (len(slots)==1) and (hard):
                #these games are fixed
                s=slots[0]
                for m in meetings:
                    #add game to solution
                    nodeid=self.nodemapping[(m[0],m[1],s)]
                    self.selectnode(nodeid)
                    #remove other possible games for these teams in this slot
                    for team in self.teams:
                        if team not in m:
                            self.deletenode(m[0], team, s)
                            self.deletenode(m[1], team, s)
                            self.deletenode(team, m[0], s)
                            self.deletenode(team, m[1], s)

        #if min<max:
        #TODO

        #if min==max <len(meetings)
        #TODO


    def addBR1(self, C):
        # TODO
        pass


    def addBR2(self, C):
        # TODO
        pass


    def addFA2(self, C):
        # TODO
        pass


    def addSE1(self, C):
        # TODO
        pass


def creategraph(path, hardconstraintcost=100):
    tree = ET.parse(path)
    root=tree.getroot()
    phased= root.find('Structure').find('Format').find('gameMode')=='P'
    resources = root.find('Resources')
    teams = [child.attrib['id'] for child in resources.find('Teams')]
    slots = [child.attrib['id'] for child in resources.find('Slots')]
    G=Graph(teams,slots,hardconstraintcost)
    idnum=0
    for s in G.slots:
        for hometeam in G.teams:
            for awayteam in G.teams:
                if awayteam!=hometeam:
                    G.nodedict[idnum]=Node(hometeam,awayteam,s,idnum)
                    G.nodemapping[(hometeam,awayteam,s)] = idnum
                    idnum+=1
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
