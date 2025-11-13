using Microsoft.AspNetCore.Mvc;
using smartloanapi.Interfaces;

namespace smartloanapi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class LoanController(ILoanService loanService) : ControllerBase
    {
        private readonly ILoanService _loanService = loanService;
    }
}