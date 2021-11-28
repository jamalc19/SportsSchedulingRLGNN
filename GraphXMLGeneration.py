import numpy as np
import random
import xml.etree.cElementTree as ET
import os

max_teams = 8

os.chdir("GenInstances/")


def pop_random(rs, team_list):
    random.seed(rs)
    i = random.randrange(0, len(team_list))
    return team_list.pop(i)


def random_init_match(rs, num_teams):
    team_list = list(range(num_teams))

    init_match = []
    while team_list:
        home = pop_random(rs, team_list)
        away = pop_random(rs, team_list)
        init_match.append([home, away])

    return init_match


def rotate_list(match_lst):
    flat_match = np.array([team for match in match_lst for team in match])
    fixed, rotate = flat_match[0], flat_match[1:]

    rotate = np.roll(rotate, 1)
    new_matches = np.insert(rotate, 0, fixed)
    new_matches_pair = [list(new_matches)[i:i+2] for i in range(0, len(new_matches), 2)]

    return new_matches_pair


def duplicate_flip(match_sched):
    rev_sched = []
    for match in match_sched:
        rev = match[::-1]
        rev_sched.append(rev)

    return rev_sched


def feasible_solution(rs):
    random.seed(rs)
    num_teams = random.randrange(4, 10, 2)
    # num_slots = (num_teams-1)*2

    match_sched = []

    init_match = random_init_match(rs, num_teams)
    match_sched.append(init_match)

    for i in range(num_teams-2):
        init_match = rotate_list(init_match)
        match_sched.append(init_match)

    second_half = []
    for i in match_sched:
        i_flip = duplicate_flip(i)
        second_half.append(i_flip)

    match_sched.extend(second_half)

    random.shuffle(match_sched)

    return match_sched, num_teams


def gen_ca1_hard(rs, num_const, sched, num_teams):
    # generates capacity constraints with max of 0 or 1, in only 1 time slot
    # can't repeat time slot/team combinations
    random.seed(rs)
    team_list = list(range(num_teams))
    slot_list = list(range((num_teams-1)*2))
    teams = []
    slots = []
    # num = []
    mode = []

    for i in range(num_const):
        violations = 0
        while True:
            tries = 0
            while True:
                t1 = random.choice(team_list)
                s1 = random.choice(slot_list)

                chosen_combo = []
                if len(teams) > 0:
                    chosen_combo = list(zip(teams, slots))

                # can only choose a team and time slot max twice, never the same team/slot combo - removed
                # if teams.count(t1) < 150 and slots.count(s1) < 100 and (t1, s1) not in chosen_combo:
                if (t1, s1) not in chosen_combo:
                    break

                tries += 1
                if tries == 100:
                    break
            if tries == 100:
                break

            # n1 = random.randrange(0,2)  # number of max - 0 or 1
            m1 = random.randrange(0, 2)  # home or away - 0 or 1

            specific_sched = sched[s1]
            violate = 0
            for match in specific_sched:
                if match[m1] == t1:
                    violate += 1

            if violate == 0:
                teams.append(t1)
                slots.append(s1)
                mode.append(m1)
                break

            violations += 1
            if violations == 100:
                break

    return [teams, slots, mode]


def gen_ca1_soft(rs, num_const, sched, num_teams, ca1_hard):
    # generates capacity constraints with max of 0 or 1, in only 1 time slot
    # can't repeat time slot/team combinations
    random.seed(rs)
    team_list = list(range(num_teams))
    slot_list = list(range((num_teams-1)*2))
    teams = []
    slots = []
    # num = []
    mode = []

    chosen_hard = list(zip(ca1_hard[0], ca1_hard[1]))
    for i in range(int(num_const)):
        violations = 0
        while True:
            tries = 0
            while True:
                t1 = random.choice(team_list)
                s1 = random.choice(slot_list)

                chosen_combo = []
                if len(teams) > 0:
                    chosen_combo = list(zip(teams, slots))

                # can only choose a team and time slot max twice, never the same team/slot combo
                if ((t1, s1) not in chosen_combo) and ((t1, s1) not in chosen_hard):
                    break

                tries += 1
                if tries == 100:
                    break

            if tries == 100:
                break

            # n1 = random.randrange(0,2)  # number of max - 0 or 1
            m1 = random.randrange(0, 2)  # home or away - 0 or 1

            specific_sched = sched[s1]
            violate = 0
            for match in specific_sched:
                if match[m1] == t1:
                    violate += 1

            if violate == 0:
                teams.append(t1)
                slots.append(s1)
                mode.append(m1)
                break

            violations += 1
            if violations == 100:
                break

    return [teams, slots, mode]


def gen_ca2(rs, num_const, sched, num_teams, ca1):
    # CA2 = <CA2 teams1="0" min="0" max="1" mode1="HA" mode2="GLOBAL" teams2="1;2" slots ="0;1;2" type="SOFT"/>
    random.seed(rs)
    team_list = list(range(num_teams))
    slot_list = list(range((num_teams - 1) * 2))
    teams1 = []
    teams2 = []
    slots = []
    num = []
    mode = []

    for i in range(num_const):
        violations = 0
        while True:
            tries = 0
            while True:
                t1 = random.choice(team_list)
                remaining_teams = [t for t in team_list if t != t1]

                t2 = sorted(random.sample(remaining_teams, random.randrange(1, 3)))
                s1 = sorted(random.sample(slot_list, random.randrange(1, 4)))

                # can only choose a team for team1 once
                if teams1.count(t1) < 1:
                    break

                tries += 1
                if tries == 20:
                    break

            if tries == 20:
                break

            n1 = random.randrange(1, 2)  # number of max - currently set at 1
            m1 = random.randrange(0, 3)  # home or away or all  - 0 or 1 or 2

            # check against valid solution
            count_times = 0
            for k in s1:
                if m1 == 0:
                    for pairs in sched[k]:
                        if pairs[0] == t1 and pairs[1] in t2:
                            count_times += 1
                if m1 == 1:
                    for pairs in sched[k]:
                        if pairs[1] == t1 and pairs[0] in t2:
                            count_times += 1
                if m1 == 2:
                    for opp in t2:
                        for pairs in sched[k]:
                            if sorted([t1, opp]) == sorted(pairs):
                                count_times += 1

            if count_times > n1:
                continue

            # check against CA1 constraints
            constraint_tuple = []
            for time in s1:
                if m1 == 0:
                    tup = (t1, time, 1)
                    constraint_tuple.append(tup)
                if m1 == 1:
                    tup = (t1, time, 0)
                    constraint_tuple.append(tup)
                if m1 == 2:
                    tup1 = (t1, time, 0)
                    tup2 = (t1, time, 1)
                    constraint_tuple.append(tup1)
                    constraint_tuple.append(tup2)

            ca1_tuple = list(zip(*ca1))

            if not (set(ca1_tuple) & set(constraint_tuple)):
                teams1.append(t1)
                teams2.append(t2)
                slots.append(s1)
                num.append(n1)
                mode.append(m1)
                break

            violations += 1
            if violations == 100:
                break

    return [teams1, teams2, slots, num, mode]


def gen_br1(rs, num_const, sched, num_teams):
    random.seed(rs)
    team_list = list(range(num_teams))
    slot_list = list(range((num_teams-1)*2))
    teams = []
    slots = []
    breaks = []
    mode = []

    for i in range(num_const):
        violations = 0
        while True:

            t1 = random.choice(team_list)
            s1 = sorted(random.sample(slot_list, random.randrange(1, 6)))  # 1-5 possible slots
            b1 = random.randrange(0, 3)  # number of max - 0 or 1 or 2
            m1 = random.randrange(0, 3)  # home or away - 0 or 1 or 2

            violate = 0
            for time in s1:
                if time != 0:
                    current_matches = sched[time]
                    previous_matches = sched[time-1]

                    for new_match in current_matches:
                        for old_match in previous_matches:
                            if (m1 == 0) or (m1 == 1):
                                if new_match[m1] == t1:
                                    if old_match[m1] == t1:
                                        violate += 1
                            if m1 == 2:
                                for home_away in range(2):
                                    if new_match[home_away] == t1:
                                        if old_match[home_away] == t1:
                                            violate += 1

            if violate <= b1:
                teams.append(t1)
                slots.append(s1)
                breaks.append(b1)
                mode.append(m1)
                break

            violations += 1
            if violations == 100:
                break

    return [teams, slots, breaks, mode]


def detect_mode(mode):
    if mode == 0:
        return "H"
    if mode == 1:
        return "A"
    if mode == 2:
        return "HA"


def detect_slots(slots):
    cslot = str(slots[0])
    for i in range(1, len(slots) - 1):
        cslot += ";"
        cslot += str(slots[i])

    if len(slots) > 1:
        cslot += ";" + str(slots[-1])

    return cslot


def gen_xml_nodes(main_node, name):
    child = ET.SubElement(main_node, name)

    return child


def gen_instances(rs):
    schedule, number_teams = feasible_solution(rs)
    number_slots = (number_teams-1)*2

    ca1_hard = gen_ca1_hard(rs, number_teams*(number_teams-1), schedule, number_teams)
    ca1_soft = gen_ca1_soft(rs, number_teams*(number_teams-1)/2, schedule, number_teams, ca1_hard)
    ca2 = gen_ca2(rs, number_teams, schedule, number_teams, ca1_hard)
    br1 = gen_br1(rs, random.randrange(number_teams, number_teams*2), schedule, number_teams)

    root = ET.Element("Instance")

    optimal = ET.SubElement(root, "OptimalSolution")
    optimal_set = [gen_xml_nodes(optimal, 'games') for i in range(len(schedule))]
    for i in range(len(optimal_set)):
        optimal_set[i].set('match', str(schedule[i]))

    structure = ET.SubElement(root, "Structure")
    ft = ET.SubElement(structure, "Format", leagueIds=str(rs))
    ET.SubElement(ft, "gameMode").text = "NP"

    resources = ET.SubElement(root, "Resources")

    leagues = ET.SubElement(resources, "Leagues")
    l1 = ET.SubElement(leagues, "league")
    l1.set('id', str(rs))
    l1.set('name', "League" + str(rs))

    teams = ET.SubElement(resources, "Teams")
    child_teams = [gen_xml_nodes(teams, 'team') for i in range(number_teams)]
    for i in range(len(child_teams)):
        child_teams[i].set('id', str(i))
        child_teams[i].set('league', str(rs))
        child_teams[i].set('name', "Team" + str(i))

    slots = ET.SubElement(resources, "Slots")
    child_slots = [gen_xml_nodes(slots, 'slot') for i in range(number_slots)]
    for i in range(len(child_slots)):
        child_slots[i].set('id', str(i))
        child_slots[i].set('name', "Slot" + str(i))

    constraints = ET.SubElement(root, "Constraints")
    capacity_constraints = ET.SubElement(constraints, "CapacityConstraints")
    ca1_constraints_h = [gen_xml_nodes(capacity_constraints, 'CA1') for i in range(len(ca1_hard[0]))]
    for i in range(len(ca1_constraints_h)):
        ca1_constraints_h[i].set('max', str(0))
        ca1_constraints_h[i].set('min', str(0))
        ca1_constraints_h[i].set('mode', detect_mode(ca1_hard[2][i]))
        ca1_constraints_h[i].set('penalty', str(1))
        ca1_constraints_h[i].set('slots', str(ca1_hard[1][i]))
        ca1_constraints_h[i].set('teams', str(ca1_hard[0][i]))
        ca1_constraints_h[i].set('type', "HARD")

    ca1_constraints_s = [gen_xml_nodes(capacity_constraints, 'CA1') for i in range(len(ca1_soft[0]))]
    for i in range(len(ca1_constraints_s)):
        ca1_constraints_s[i].set('max', str(0))
        ca1_constraints_s[i].set('min', str(0))
        ca1_constraints_s[i].set('mode', detect_mode(ca1_soft[2][i]))
        ca1_constraints_s[i].set('penalty', str(5))
        ca1_constraints_s[i].set('slots', str(ca1_soft[1][i]))
        ca1_constraints_s[i].set('teams', str(ca1_soft[0][i]))
        ca1_constraints_s[i].set('type', "SOFT")

    ca2_constraints = [gen_xml_nodes(capacity_constraints, 'CA2') for i in range(len(ca2[0]))]
    for i in range(len(ca2_constraints)):
        ca2_constraints[i].set('max', str(ca2[3][i]))
        ca2_constraints[i].set('min', str(0))
        ca2_constraints[i].set('mode1', detect_mode(ca2[4][i]))
        ca2_constraints[i].set('mode2', 'GLOBAL')
        ca2_constraints[i].set('penalty', str(1))
        ca2_constraints[i].set('slots', detect_slots(ca2[2][i]))
        ca2_constraints[i].set('teams1', str(ca2[0][i]))
        ca2_constraints[i].set('teams2', detect_slots(ca2[1][i]))
        ca2_constraints[i].set('type', "HARD")

    break_constraints = ET.SubElement(constraints, "BreakConstraints")
    br1_constraints = [gen_xml_nodes(break_constraints, 'BR1') for i in range(len(br1[0]))]
    for i in range(len(br1_constraints)):
        br1_constraints[i].set('intp', str(br1[2][i]))
        br1_constraints[i].set('mode1', "LEQ")
        br1_constraints[i].set('mode2', detect_mode(br1[3][i]))
        br1_constraints[i].set('penalty', str(5))
        br1_constraints[i].set('slots', detect_slots(br1[1][i]))
        br1_constraints[i].set('teams', str(br1[0][i]))
        br1_constraints[i].set('type', "SOFT")

    tree = ET.ElementTree(root)
    tree.write("gen_instance_" + str(rs) + ".xml")


for instances in range(100):
    gen_instances(instances)


