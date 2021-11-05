class Node:
    def __init__(self,hometeam,awayteam,slot,idnum,selected=False, cost=0):
        self.hometeam=hometeam
        self.awayteam=awayteam
        self.slot=slot
        self.id=idnum
        self.selected=selected
        self.cost=cost
        #edges indexed by the id of the node at the other end
        self.edges_soft={} #edges that represent soft constraints
        self.edges_hard={} #edges that represent hard constraints
        self.edges_soft_complex={} #edges that approximately represent soft constraints that relate more than 2 games
        self.edges_hard_complex={} #edges that approximately represent hard constraints that relate more than 2 games

    def delete(self):
        for edge in self.edges_soft:
            edge.delete()
        for edge in self.edges_hard:
            edge.delete()
        for edge in self.edges_soft_complex:
            edge.delete()
        for edge in self.edges_hard_complex:
            edge.delete()
        #have to remove from nodedict as well

    def addcost(self,cost):
        self.cost+=cost

class Edge:
    def __init__(self,node1,node2,weight,hard=False,Complex=False, complexid=None):
        #node1.idnum < node2.idnum
        self.node1=node1
        self.node2=node2
        self.weight=weight
        self.hard=hard
        self.complex = Complex
        self.complexid=complexid
        if self.hard:
            if self.complex:
                self.node1.edges_hard_complex[self.node2.id]=self
                self.node2.edges_hard_complex[self.node1.id]=self
            else:
                self.node1.edges_hard[self.node2.id]=self
                self.node2.edges_hard[self.node1.id]=self
        else:
            if self.complex:
                self.node1.edges_soft_complex[self.node2.id]=self
                self.node2.edges_soft_complex[self.node1.id]=self
            else:
                self.node1.edges_soft[self.node2.id]=self
                self.node2.edges_soft[self.node1.id]=self

    def delete(self):
        if self.hard:
            if self.complex:
                del self.node1.edges_hard_complex[self.node2.id]
                del self.node2.edges_hard_complex[self.node1.id]
            else:
                del self.node1.edges_hard[self.node2.id]
                del self.node2.edges_hard[self.node1.id]
        else:
            if self.complex:
                del self.node1.edges_soft_complex[self.node2.id]
                del self.node2.edges_soft_complex[self.node1.id]
            else:
                del self.node1.edges_soft[self.node2.id]
                del self.node2.edges_soft[self.node1.id]


    
