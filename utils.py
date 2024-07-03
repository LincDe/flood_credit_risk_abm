import numpy as np

def density_preprocess(map):
    '''
    input: density map
    output: 2 lists
    1. 'density_slots' contains the cumulated probability/density, 
    2. 'coordinates' contains the corresponding coordinates
    '''
    map = np.array(map)
    n, m = np.shape(map)
    sum = 0
    density_slots = []
    coordinates = []
    for i in range(n):
        for j in range(m):
            if map[i][j] > 0:
                sum += map[i][j]
                density_slots.append(sum)
                coordinates.append([i, j])
    # normalize density_slots
    density_slots = density_slots / sum
    return density_slots, coordinates

# density_preprocess(map)
def find_coordi(rv, density_slots, coordinates):
    '''
    input: a random number 'rv', cumulated density (sorted list), corrsponding coordinates (list)
    output: the coordinates according to the random number
    '''
    # binary search
    # high, low, mid here are indexes, not values
    low = 0
    high = len(density_slots)

    while low<=high:
        mid = (low + high) // 2 
        if rv < density_slots[mid]:
            high = mid -1
        elif rv > density_slots[mid]:
            low = mid +1
        else:
            return coordinates[mid]
    # print(high, mid, low)
    if mid == low:
        return coordinates[mid]
    else:
        return coordinates[low]
    

    # Function to calculate EMI
def calculate_emi(principal, annual_interest_rate, loan_tenure_years = 10):
    r = annual_interest_rate / (12 * 100)  # Monthly interest rate
    n = loan_tenure_years * 12  # Total number of monthly installments
    emi = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
    return emi

# score card
def scorecard(r_cap, income, sen, expenditure, fund, ltv, install, v, sp, tm):
    # TODO: finish the nj score card part
    # TODO: use this score to initialize agents
    """compute pd score for new joiners"""
    score = 0
    # ltv
    if ltv <= 0.4:
        score += 20
    elif ltv <=0.7:
        score += 14
    elif ltv <= 0.9:
        score += 7

    # income
    if income <= 300:
        score += 12
    elif income <= 500:
        score += 17
    elif income <=800:
        score += 24
    elif income <= 1700:
        score += 34
    else:
        score += 38

    # seniority
    if sen <= 15:
        score += 9
    elif sen <= 47:
        score += 14
    else:
        score += 28

    # r_cap
    if r_cap <=0.05:
        score += 9
    elif r_cap <= 0.40:
        score += 22
    elif r_cap <= 0.50:
        score += 24
    else:
        score += 33
    
    # r_inst
    r_inst = install / income
    if r_inst <= 0.40:
        score += 20
    else:
        score += 9
    
    # credit score
    # ratio_arr = np.sum(self.v_arr) / 36
    # score += 100 * (1 - ratio_arr)
    score += 100

    return score