
#include <cassert>
#include <iostream>
#include <cstdlib>

#include "bluff.h"

using namespace std; 

static unsigned long long nextReport = 1;
static unsigned long long reportMult = 2;

// This is Vanilla CFR. See my thesis, Algorithm 1 (Section 2.2.2)
double cfr(GameState & gs, int player, int depth, unsigned long long bidseq, 
           double reach1, double reach2, double chanceReach, int phase, int updatePlayer)
{
  // at terminal node?
  if (terminal(gs))
  {
    return payoff(gs, updatePlayer);
  }

  nodesTouched++;

  // Chances nodes at the top of the tree. If p1roll and p2roll not set, we're at a chance node
  if (gs.p1roll == 0) 
  {
    double EV = 0.0; 

    for (int i = 1; i <= numChanceOutcomes(1); i++) 
    {
      GameState ngs = gs; 
      ngs.p1roll = i; 
      double newChanceReach = getChanceProb(1,i)*chanceReach;

      EV += getChanceProb(1,i)*cfr(ngs, player, depth+1, bidseq, reach1, reach2, newChanceReach, phase, updatePlayer); 
    }

    return EV;
  }
  else if (gs.p2roll == 0)
  {
    double EV = 0.0; 

    for (int i = 1; i <= numChanceOutcomes(2); i++)
    {
      GameState ngs = gs; 
      ngs.p2roll = i; 
      double newChanceReach = getChanceProb(2,i)*chanceReach;

      EV += getChanceProb(2,i)*cfr(ngs, player, depth+1, bidseq, reach1, reach2, newChanceReach, phase, updatePlayer); 
    }

    return EV;
  }

  // Check for cuts. This is the pruning optimization described in Section 2.2.2 of my thesis. 
  if (phase == 1 && (   (player == 1 && updatePlayer == 1 && reach2 <= 0.0)
                     || (player == 2 && updatePlayer == 2 && reach1 <= 0.0)))
  {
    phase = 2; 
  }

  if (phase == 2 && (   (player == 1 && updatePlayer == 1 && reach1 <= 0.0)
                     || (player == 2 && updatePlayer == 2 && reach2 <= 0.0)))
  {
    return 0.0;
  }

  // declare the variables 
  Infoset is;
  unsigned long long infosetkey = 0;
  double stratEV = 0.0;
  int action = -1;

  int maxBid = (gs.curbid == 0 ? BLUFFBID-1 : BLUFFBID);
  int actionshere = maxBid - gs.curbid; 

  assert(actionshere > 0);
  double moveEVs[actionshere]; 
  for (int i = 0; i < actionshere; i++) 
    moveEVs[i] = 0.0;

  // get the info set (also set is.curMoveProbs using regret matching)
  getInfoset(gs, player, bidseq, is, infosetkey, actionshere); 

  // iterate over the actions
  for (int i = gs.curbid+1; i <= maxBid; i++) 
  {
    // there is a valid action here
    action++;
    assert(action < actionshere);

    unsigned long long newbidseq = bidseq;
    double moveProb = is.curMoveProbs[action]; 
    double newreach1 = (player == 1 ? moveProb*reach1 : reach1); 
    double newreach2 = (player == 2 ? moveProb*reach2 : reach2); 

    GameState ngs = gs; 
    ngs.prevbid = gs.curbid;
    ngs.curbid = i; 
    ngs.callingPlayer = player;
    newbidseq |= (1ULL << (BLUFFBID-i)); 
    
    double payoff = cfr(ngs, 3-player, depth+1, newbidseq, newreach1, newreach2, chanceReach, phase, updatePlayer); 
   
    moveEVs[action] = payoff; 
    stratEV += moveProb*payoff; 
  }

  // post-traversals: update the infoset
  double myreach = (player == 1 ? reach1 : reach2); 
  double oppreach = (player == 1 ? reach2 : reach1); 

  if (phase == 1 && player == updatePlayer) // regrets
  {
    for (int a = 0; a < actionshere; a++)
    {
      // Multiplying by chanceReach here is important in games that have non-uniform chance outcome 
      // distributions. In Bluff(1,1) it is actually not needed, but in general it is needed (e.g. 
      // in Bluff(2,1)). 
      is.cfr[a] += (chanceReach*oppreach)*(moveEVs[a] - stratEV); 
    }
  }

  if (phase >= 1 && player == updatePlayer) // av. strat
  {
    for (int a = 0; a < actionshere; a++)
    {
      is.totalMoveProbs[a] += myreach*is.curMoveProbs[a]; 
    }
  }

  // save the infoset back to the store if needed
  if (player == updatePlayer) {
    iss.put(infosetkey, is, actionshere, 0); 
  }

  return stratEV;
}

int main(int argc, char ** argv)
{
  unsigned long long maxIters = 0; 
  init();

  if (argc < 2)
  {
    initInfosets();
    exit(-1);
  }
  else 
  { 
    string filename = argv[1];
    cout << "Reading the infosets from " << filename << "..." << endl;
    iss.readFromDisk(filename);

    if (argc >= 3)
      maxIters = to_ull(argv[2]);
  }  
  
  // get the iteration
  string filename = argv[1];
  vector<string> parts; 
  split(parts, filename, '.'); 
  if (parts.size() != 3 || parts[1] == "initial")
    iter = 1; 
  else
    iter = to_ull(parts[1]); 
  cout << "Set iteration to " << iter << endl;
  iter = MAX(1,iter);

  unsigned long long bidseq = 0; 
    
  StopWatch stopwatch;
  double totaltime = 0; 

  cout << "Starting CFR iterations" << endl;

  for (; true; iter++)
  {
    GameState gs1; 
    bidseq = 0; 
    double ev1 = cfr(gs1, 1, 0, bidseq, 1.0, 1.0, 1.0, 1, 1);
    
    GameState gs2; 
    bidseq = 0; 
    double ev2 = cfr(gs2, 1, 0, bidseq, 1.0, 1.0, 1.0, 1, 2);

    if (iter % 10 == 0)
    { 
      cout << "."; cout.flush(); 
      totaltime += stopwatch.stop();
      stopwatch.reset();
    }

    if (iter == 1 || nodesTouched >= ntNextReport)
    {
      cout << endl;

      cout << "total time: " << totaltime << " seconds." << endl; 
      cout << "Done iteration " << iter << endl;

      cout << "ev1 = " << ev1 << ", ev2 = " << ev2 << endl;

      // This bound is the right-hand side of Theorem 3 from the original CFR paper.
      // \sum_{I \in \II_i} R_{i,imm}^{T,+}(I)

      cout << "Computing bounds... "; cout.flush(); 
      double b1 = 0.0, b2 = 0.0;
      iss.computeBound(b1, b2); 
      cout << " b1 = " << b1 << ", b2 = " << b2 << ", bound = " << (2.0*MAX(b1,b2)) << endl;

      double conv = 0.0;
      double p1value = 0.0;
      double p2value = 0.0;
      conv = computeBestResponses(false, p1value, p2value);

      report("cfr.bluff11.report.txt", totaltime, (2.0*MAX(b1,b2)), conv);
      //dumpInfosets("iss");

      cout << endl;

      nextCheckpoint += cpWidth;
      nextReport *= reportMult;
      ntNextReport *= ntMultiplier;

      stopwatch.reset(); 
    }

    if (iter == maxIters) break;
  }
}

