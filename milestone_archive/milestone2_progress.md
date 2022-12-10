For milestone 2, Evan has been assigned the implementation of forward mode autodiff for scalar functions of scalar inputs. This implementation will rely on python dictionaries to house computational graphs which should be sufficient for milestone 2 and our planned extension. In addition, Evan will provide support for and contribute to updating and extending our package's documentation with the specifics of scalar function autodiff functionality. No progress has been made at the time of writing.

Luke will first work on writing code for overloading existing operators (__add__, __sub__, __mul__, etc), as well as defining new functions for other necessary operations in computing derivatives (sqrt, log, exp, sin, cos, etc). These functions can then passed to the forward mode implementation written by Evan.

Kane is working with Evan on the forward mode autodiff. Specifically, he is making sure Luke and Evan's work is integrated. Additionally, he will handle updating and extending our documentation for milestone 2.

Andrew is working on CI/CD, and will support Kane with documentation. In addition to this, he will be running the user acceptence testing and helping the team track their progress towards their deliverables. 

James is supporting the team with the testing, and making sure we operate a "black box" testing protocol. This is particularly relevant for the next milestone, where a working test suite needs to be implemented. He will also provide support where needed in other domains, particularly with CI/CD. 
