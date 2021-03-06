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
        for edge in set(self.edges_soft.keys()):
            self.edges_soft[edge].delete()
        for edge in set(self.edges_hard.keys()):
            self.edges_hard[edge].delete()
        for edge in set(self.edges_soft_complex.keys()):
            self.edges_soft_complex[edge].delete()
        for edge in set(self.edges_hard_complex.keys()):
            self.edges_hard_complex[edge].delete()

    def addcost(self,cost):
        self.cost+=cost



class Edge:
    def __init__(self,node1,node2,weight,hard=False,Complex=False, constraintid=None):
        #node1.idnum < node2.idnum
        if node1==node2:
            print('ERROR: self directed edge')
        self.node1=node1
        self.node2=node2
        self.weight=weight
        self.hard=hard
        self.complex = Complex
        #only constraints that require updating throughout the solving process require an id
        self.constraintids=set()
        if constraintid is not None:
            self.constraintids.add(constraintid)
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

    def __hash__(self):
        return (self.node1.id, self.node2.id,self.hard,self.complex).__hash__()

    def addweight(self,weight):
        self.weight+=weight

    def deleteconstraint(self, C_id, weight):
        self.constraintids.discard(C_id)
        if len(self.constraintids) == 0:
            self.delete()
        else:
            self.addweight(-weight)

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


    
